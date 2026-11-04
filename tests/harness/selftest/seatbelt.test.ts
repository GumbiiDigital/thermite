import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, test } from "node:test";
import { SeatbeltRunner } from "../../../src/harness/broker/seatbelt.ts";
import { isolationSummary } from "../../../src/harness/selftest/index.ts";
import { SEATBELT_GAPS, seatbeltSelftest } from "../../../src/harness/selftest/seatbelt.ts";

const onMac = {
  skip: process.platform === "darwin" ? false : "Seatbelt is macOS's: this runs on a Mac",
};

describe("the macOS isolation selftest", () => {
  test(
    "every role's probes come out as the fence promises, and the gaps are listed",
    onMac,
    async () => {
      const root = realpathSync(mkdtempSync(join(tmpdir(), "thermite-selftest-")));
      const stateDir = join(root, "state");
      const scratchDir = join(root, "scratch");
      mkdirSync(stateDir);
      mkdirSync(scratchDir);
      const runner = new SeatbeltRunner({ stateDir, scratchDir, env: {} });
      const report = await seatbeltSelftest({ runner, stateDir, scratchDir });
      assert.equal(report.ok, true, isolationSummary(report));
      assert.deepEqual(Object.keys(report.roles), [
        "builder",
        "run",
        "checker",
        "driver",
        "lane-1",
        "lane-1 on the GPU",
        "agent-1",
        "planner",
      ]);
      const r = report.roles;
      assert.equal(r.run?.gpu, "opened");
      assert.equal(r["lane-1 on the GPU"]?.gpu, "opened");
      assert.equal(r.builder?.gpu, "refused");
      assert.equal(r.checker?.["read-kit"], "readable");
      assert.equal(r.run?.["read-kit"], "refused");
      assert.equal(r.driver?.network, "refused");
      assert.equal(r["agent-1"]?.network, "reachable");
      for (const probes of Object.values(r)) {
        assert.equal(probes["read-state"], "refused");
        assert.equal(probes["read-other-role"], "refused");
        assert.equal(probes.signal, "refused");
      }
      assert.deepEqual(report.skipped, SEATBELT_GAPS);
      assert.match(isolationSummary(report), /^isolation holds/);
    },
  );
});
