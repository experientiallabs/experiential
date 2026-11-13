//! Native transaction checks for shared and destination budget scope precedence.

use rusqlite::{params, Connection};

use super::budgets::{self, Reservation};

fn database() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(
        "CREATE TABLE gateway_monthly_budgets (
        budget_id TEXT PRIMARY KEY, organization_id TEXT, period_start TEXT,
        scope_kind TEXT, scope_key TEXT, identity_id TEXT, alias_id TEXT, pool_id TEXT,
        deployment_id TEXT, limit_nano_usd INTEGER, reserved_nano_usd INTEGER DEFAULT 0,
        settled_nano_usd INTEGER DEFAULT 0, strict_unknown_cost INTEGER DEFAULT 0,
        unknown_cost_attempts INTEGER DEFAULT 0);
        CREATE TABLE gateway_attempt_budget_charges (
        attempt_id TEXT,budget_id TEXT,reserved_nano_usd INTEGER,settled_nano_usd INTEGER,
        PRIMARY KEY(attempt_id,budget_id));",
    )
    .unwrap();
    db
}

fn reservation() -> Reservation<'static> {
    Reservation {
        organization: "org",
        identity: "identity",
        alias: "alias",
        root_pool: "root",
        pool: "child",
        deployment: "child-1",
        attempt: "attempt",
        period: "2026-08",
        maximum: Some(38),
    }
}

fn budget(db: &Connection, id: &str, kind: &str, pool: Option<&str>, limit: i64) {
    db.execute("INSERT INTO gateway_monthly_budgets
        (budget_id,organization_id,period_start,scope_kind,scope_key,identity_id,alias_id,pool_id,limit_nano_usd)
        VALUES (?,'org','2026-08',?,?,'identity','alias',?,?)",params![id,kind,id,pool,limit]).unwrap();
}

#[test]
fn shared_and_root_denials_precede_child_denials_without_partial_holds() {
    for kind in ["root", "team", "identity", "unknown"] {
        for child in ["000-child", "zzz-child"] {
            let mut db = database();
            budget(
                &db,
                "parent",
                if kind == "root" {
                    "pool"
                } else if kind == "unknown" {
                    "team"
                } else {
                    kind
                },
                Some("root"),
                if kind == "unknown" { 1000 } else { 0 },
            );
            budget(&db, child, "pool", Some("child"), 0);
            if kind == "unknown" {
                db.execute("UPDATE gateway_monthly_budgets SET strict_unknown_cost=1,unknown_cost_attempts=1 WHERE budget_id='parent'",[]).unwrap();
            }
            let tx = db.transaction().unwrap();
            let refusal = budgets::reserve(&tx, &reservation()).unwrap().unwrap();
            assert_eq!(refusal["scope_kind"], "request");
            assert_eq!(
                refusal["application"],
                if kind == "root" { "root" } else { "shared" }
            );
            if kind == "unknown" {
                assert!(refusal["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("prior attempts with unknown cost"));
            }
            tx.commit().unwrap();
            assert_eq!(
                db.query_row(
                    "SELECT count(*) FROM gateway_attempt_budget_charges",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                0
            );
            assert_eq!(
                db.query_row(
                    "SELECT sum(reserved_nano_usd) FROM gateway_monthly_budgets",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                0
            );
        }
    }
}

#[test]
fn destination_refusal_is_scoped_and_successful_holds_settle_exactly_once() {
    let mut db = database();
    budget(&db, "parent", "pool", Some("root"), 1000);
    budget(&db, "child", "pool", Some("child"), 0);
    let tx = db.transaction().unwrap();
    let refusal = budgets::reserve(&tx, &reservation()).unwrap().unwrap();
    assert_eq!(refusal["scope_kind"], "pool");
    assert_eq!(refusal["scope_id"], "child");
    assert_eq!(refusal["application"], "destination");
    assert_eq!(refusal["budget_scope"]["alias_id"], "alias");
    tx.commit().unwrap();
    db.execute("UPDATE gateway_monthly_budgets SET limit_nano_usd=1000", [])
        .unwrap();
    let tx = db.transaction().unwrap();
    assert!(budgets::reserve(&tx, &reservation()).unwrap().is_none());
    tx.commit().unwrap();
    assert_eq!(
        db.query_row(
            "SELECT sum(reserved_nano_usd) FROM gateway_monthly_budgets",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        76
    );
    for _ in 0..2 {
        let tx = db.transaction().unwrap();
        budgets::settle(&tx, "attempt", Some(19)).unwrap();
        tx.commit().unwrap();
    }
    assert_eq!(
        db.query_row(
            "SELECT sum(reserved_nano_usd),sum(settled_nano_usd) FROM gateway_monthly_budgets",
            [],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
        )
        .unwrap(),
        (0, 38)
    );
}
