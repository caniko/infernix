{
  lib,
  package,
  keyFile,
}: ''
  key="$(${package}/bin/infernix-codex-credentials read ${lib.escapeShellArg keyFile})"
  export INFERNIX_CODEX_PROVIDER_API_KEY="$key"
''
