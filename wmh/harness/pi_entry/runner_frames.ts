/**
 * Length-prefixed JSON frame client for the RunnerLink transport (the Node peer of
 * wmh/harness/runner_link.py). One TCP socket to the host carries every episode; requests get a
 * fresh `req_id` and resolve when the matching response frame arrives, while server-pushed frames
 * (episode_start, cancel, ping) fire registered handlers. Wire format matches runner_link.py
 * exactly: 4-byte big-endian length prefix + UTF-8 JSON body.
 */
import net from "node:net";

export type Frame = Record<string, any>;

export class FrameConn {
	private sock: net.Socket;
	private buf: Buffer = Buffer.alloc(0);
	private waiters = new Map<number, (f: Frame) => void>();
	private handlers = new Map<string, (f: Frame) => void>();
	private reqSeq = 0;
	private closed = false;

	constructor(sock: net.Socket) {
		this.sock = sock;
		sock.on("data", (d: Buffer) => this._onData(d));
		// If the channel drops while a request is in flight, settle every pending waiter with an
		// error frame — otherwise awaiting llm_request/tool_request promises hang forever and the
		// episode never returns a done/episode_error.
		sock.on("close", () => this._settleAll("runner channel closed"));
		sock.on("error", (e: Error) => this._settleAll(`runner channel error: ${e.message}`));
	}

	private _settleAll(reason: string): void {
		this.closed = true;
		const pending = [...this.waiters.values()];
		this.waiters.clear();
		for (const resolve of pending) {
			resolve({ error: reason, content: reason, is_error: true });
		}
	}

	/** Register a handler for a server-pushed frame type (no req_id): episode_start, cancel, ping. */
	on(type: string, handler: (f: Frame) => void): void {
		this.handlers.set(type, handler);
	}

	send(frame: Frame): void {
		const body = Buffer.from(JSON.stringify(frame), "utf8");
		const hdr = Buffer.alloc(4);
		hdr.writeUInt32BE(body.length, 0);
		this.sock.write(Buffer.concat([hdr, body]));
	}

	/** Send a request frame with a fresh req_id; resolve with the matching response frame. */
	request(type: string, payload: Frame): Promise<Frame> {
		if (this.closed) {
			const reason = "runner channel closed";
			return Promise.resolve({ error: reason, content: reason, is_error: true });
		}
		const req_id = ++this.reqSeq;
		return new Promise((resolve) => {
			this.waiters.set(req_id, resolve);
			this.send({ type, req_id, ...payload });
		});
	}

	private _onData(chunk: Buffer): void {
		this.buf = Buffer.concat([this.buf, chunk]);
		while (this.buf.length >= 4) {
			const n = this.buf.readUInt32BE(0);
			if (this.buf.length < 4 + n) break;
			const body = this.buf.subarray(4, 4 + n);
			this.buf = this.buf.subarray(4 + n);
			let frame: Frame;
			try {
				frame = JSON.parse(body.toString("utf8"));
			} catch {
				continue;
			}
			this._dispatch(frame);
		}
	}

	private _dispatch(frame: Frame): void {
		const rid = frame.req_id;
		if (typeof rid === "number" && this.waiters.has(rid)) {
			const resolve = this.waiters.get(rid);
			this.waiters.delete(rid);
			resolve?.(frame);
			return;
		}
		const handler = this.handlers.get(frame.type);
		if (handler) handler(frame);
	}
}
