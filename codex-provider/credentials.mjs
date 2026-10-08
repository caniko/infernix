import { randomBytes } from "node:crypto";
import { closeSync, constants, fstatSync, fsyncSync, lstatSync, mkdirSync, openSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";

export function readKeyFile(path) {
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.uid !== process.getuid() || (stat.mode & 0o777) !== 0o600 || stat.nlink !== 1 || stat.size > 128) {
      throw new Error("Codex credential must be an owned, private regular file (0600)");
    }
    const key = readFileSync(fd, "utf8").trim();
    if (!/^[a-f0-9]{64}$/.test(key)) throw new Error("Codex credential must contain a 256-bit random key");
    return key;
  } finally {
    closeSync(fd);
  }
}

export function ensureKeyFile(path) {
  const directory = dirname(path);
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  const stat = lstatSync(directory);
  if (!stat.isDirectory() || stat.uid !== process.getuid() || (stat.mode & 0o777) !== 0o700) {
    throw new Error("Codex credential directory must be owned and private (0700)");
  }
  let fd;
  try {
    fd = openSync(path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    return readKeyFile(path);
  }
  try {
    writeFileSync(fd, `${randomBytes(32).toString("hex")}\n`);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
  return readKeyFile(path);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [action, path] = process.argv.slice(2);
  if (!path || !["ensure", "read"].includes(action)) throw new Error("usage: infernix-codex-credentials ensure|read KEY_FILE");
  if (action === "ensure") ensureKeyFile(path);
  else process.stdout.write(readKeyFile(path));
}
