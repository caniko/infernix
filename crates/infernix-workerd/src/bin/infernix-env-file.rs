//! Render a systemd `EnvironmentFile` from literals and secret files.
//!
//! The unit owns identity: it runs this with `Group=` and a
//! `RuntimeDirectory=`, so the file lands group-readable without a chown.

use std::fs::{self, OpenOptions, Permissions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;

#[derive(Parser)]
#[command(
    name = "infernix-env-file",
    about = "Render a systemd environment file"
)]
struct Args {
    /// Destination environment file to atomically replace (mode 0640).
    #[arg(long = "output-path")]
    output: PathBuf,

    /// Non-secret `KEY=VALUE` entry; written before the secrets.
    #[arg(long = "literal", value_name = "KEY=VALUE")]
    literals: Vec<String>,

    /// `KEY=PATH` entry whose value is read from a secret file.
    #[arg(long = "secret", value_name = "KEY=PATH")]
    secrets: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    render(&args.output, &args.literals, &args.secrets)
}

fn split(entry: &str) -> Result<(&str, &str)> {
    entry
        .split_once('=')
        .filter(|(key, _)| {
            !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
        .ok_or_else(|| anyhow!("expected KEY=VALUE with a variable name before '='"))
}

/// Errors name the key and path only; a secret value never reaches them.
fn render(output: &Path, literals: &[String], secrets: &[String]) -> Result<()> {
    let mut rendered = String::new();
    for entry in literals {
        let (key, value) = split(entry)?;
        if value.contains(['\r', '\n']) {
            bail!("literal {key} contains a line break");
        }
        rendered.push_str(&format!("{key}={value}\n"));
    }
    for entry in secrets {
        let (key, path) = split(entry)?;
        let value = fs::read_to_string(path)
            .with_context(|| format!("missing readable secret for {key} at {path}"))?;
        let value: String = value
            .chars()
            .filter(|c| !matches!(c, '\r' | '\n'))
            .collect();
        if value.is_empty() {
            bail!("empty secret for {key} at {path}");
        }
        rendered.push_str(&format!("{key}={value}\n"));
    }

    // One writer per unit, so a fixed sibling name is enough; a leftover from
    // a killed run is truncated and reused.
    let mut tmp = output.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)
        .with_context(|| format!("create {}", tmp.display()))?;
    file.set_permissions(Permissions::from_mode(0o600))?;
    file.write_all(rendered.as_bytes())?;
    file.set_permissions(Permissions::from_mode(0o640))?;
    file.sync_all()?;
    fs::rename(&tmp, output).with_context(|| format!("replace {}", output.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn renders_literals_then_secrets_and_refuses_bad_secrets() {
        let dir = std::env::temp_dir().join(format!("infernix-env-file-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let secret = dir.join("secret");
        let empty = dir.join("empty");
        fs::write(&secret, "s3cr=et\r\n").unwrap();
        fs::write(&empty, "\n").unwrap();
        let out = dir.join("env");
        let arg = |key: &str, path: &Path| format!("{key}={}", path.display());
        let expected = "A=b=c\nEMPTY=\nKEY=s3cr=et\n";

        render(
            &out,
            &["A=b=c".to_string(), "EMPTY=".to_string()],
            &[arg("KEY", &secret)],
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&out).unwrap(), expected);
        assert_eq!(fs::metadata(&out).unwrap().mode() & 0o7777, 0o640);

        // Failures keep the previous file, leave no temporary file, and never
        // quote the value of a secret that was already read.
        for bad in [arg("KEY", &empty), arg("KEY", &dir.join("absent"))] {
            let err = render(&out, &[], &[arg("OK", &secret), bad]).unwrap_err();
            assert!(!format!("{err:#}").contains("s3cr"));
        }
        assert!(render(&out, &["bad key=1".to_string()], &[]).is_err());
        assert!(render(&out, &["A=1\nB=2".to_string()], &[]).is_err());
        assert_eq!(fs::read_to_string(&out).unwrap(), expected);
        assert!(!dir.join("env.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
