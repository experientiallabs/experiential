"""Tests for interactive Bedrock credential-mode inference."""

from exp.cli.providers.bedrock_credentials import infer_bedrock_auth


def test_temporary_aws_credentials_stay_on_the_ambient_chain() -> None:
    """A three-part STS credential is never truncated into explicit pair mode."""
    assert infer_bedrock_auth(
        {
            "AWS_ACCESS_KEY_ID": "AKIAEXAMPLEKEY0001",
            "AWS_SECRET_ACCESS_KEY": "temporary-secret",
            "AWS_SESSION_TOKEN": "temporary-session-token",
        }
    ) == (None, None, None)


def test_long_lived_aws_pair_uses_explicit_locators() -> None:
    """A two-part access-key pair remains eligible for explicit authority."""
    assert infer_bedrock_auth(
        {
            "AWS_ACCESS_KEY_ID": "AKIAEXAMPLEKEY0001",
            "AWS_SECRET_ACCESS_KEY": "long-lived-secret",
        }
    ) == ("AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "access_key_pair")
