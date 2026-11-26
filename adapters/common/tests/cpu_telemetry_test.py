import resource
from types import SimpleNamespace
from unittest import mock

import adapter_test


class CpuTelemetryTests(adapter_test.Harness):
    def test_reports_parent_and_child_cpu_delta_and_peak_rss(self):
        def usage(user, system, rss):
            return SimpleNamespace(ru_utime=user, ru_stime=system, ru_maxrss=rss)

        stats = {
            resource.RUSAGE_SELF: usage(1.0, 0.5, 1024),
            resource.RUSAGE_CHILDREN: usage(2.0, 0.25, 4096),
        }

        def handler(ctx):
            stats[resource.RUSAGE_SELF] = usage(1.25, 0.75, 2048)
            stats[resource.RUSAGE_CHILDREN] = usage(3.0, 0.75, 8192)
            (ctx.output_directory / "output.bin").write_bytes(b"output")
            return [("data", "output.bin", "application/octet-stream")]

        profile = {**adapter_test.PROFILE, "backend": "cpu", "device": "cpu", "cpuPolicy": "permitted"}
        document = adapter_test.envelope(self.staging, profile=profile)
        with mock.patch.object(resource, "getrusage", side_effect=stats.__getitem__):
            code = self.execute(handler, document)
        self.assertEqual(code, 0, self.stderr)
        summary = self.payload()["telemetry"]
        self.assertEqual(summary["cpuSeconds"], 2.0)
        self.assertEqual(summary["peakRssBytes"], 8192 * 1024)
