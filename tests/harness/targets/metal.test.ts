import assert from "node:assert/strict";
import { describe, test } from "node:test";
import {
  MetalDriver,
  parseAccelerator,
  thermalWarnings,
} from "../../../src/harness/targets/metal.ts";

// ioreg, pmset, sysctl and sw_vers output shaped like a Mac with Apple silicon (no real host data).
const ACCELERATOR = (utilization: number) => `+-o AGXAcceleratorG16X  <class AGXAcceleratorG16X>
    {
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=5138268160,"Tiler Utilization %"=1,"Renderer Utilization %"=${utilization},"Device Utilization %"=${utilization},"In use system memory"=2217279488}
      "model" = "Apple M9 Test"
      "gpu-core-count" = 40
    }
`;
const PLATFORM = `+-o J000AP  <class IOPlatformExpertDevice>
    {
      "IOPlatformUUID" = "00000000-1111-2222-3333-444444444444"
    }
`;
const CALM = `Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
`;
const HOT = `Thermal warning level set to 2.
Performance warning level set to 0.
 CPU_Speed_Limit 	= 70
`;

function fake(utilization: number, therm = CALM) {
  const seen: string[] = [];
  const exec = async (cmd: string, args: string[]) => {
    seen.push(`${cmd} ${args.join(" ")}`);
    if (cmd === "ioreg" && args.includes("IOAccelerator")) return ACCELERATOR(utilization);
    if (cmd === "ioreg") return PLATFORM;
    if (cmd === "pmset") return therm;
    if (cmd === "sysctl") return "Apple M9 Test\n";
    if (cmd === "sw_vers") return args[0] === "-productVersion" ? "99.1\n" : "99A100\n";
    throw new Error(`unexpected ${cmd}`);
  };
  return { exec, seen };
}

describe("metal-gpu driver", () => {
  test("parses the GPU's fields from ioreg, not the driver's own memory field", () => {
    assert.deepEqual(parseAccelerator(ACCELERATOR(3)), {
      model: "Apple M9 Test",
      cores: 40,
      utilization: 3,
      memoryInUse: 2217279488,
    });
    assert.equal(parseAccelerator("nothing").utilization, undefined);
  });

  test("pmset's notes are no warning; a level above 0 or a speed limit is", () => {
    assert.deepEqual(thermalWarnings(CALM), []);
    assert.deepEqual(thermalWarnings("Thermal warning level set to 0.\n"), []);
    assert.deepEqual(thermalWarnings(HOT), [
      "thermal warning: Thermal warning level set to 2.",
      "CPU speed limited to 70%",
    ]);
  });

  test("an idle, calm GPU is ready; its identity is the Mac's platform UUID", async () => {
    const { exec, seen } = fake(2);
    const d = new MetalDriver({ targetClass: "m9-40c", exec });
    assert.deepEqual(await d.readyForTiming(), {
      ready: true,
      reasons: [],
      details: { utilization: 2, memory_in_use: 2217279488, warnings: [] },
    });
    const id = await d.identify();
    assert.equal(id.type, "metal-gpu");
    assert.equal(id.class, "m9-40c");
    assert.equal(id.id, "00000000-1111-2222-3333-444444444444");
    assert.deepEqual(id.details, { chip: "Apple M9 Test", gpu: "Apple M9 Test", gpu_cores: 40 });
    assert.ok(seen.includes("ioreg -rd1 -c IOPlatformExpertDevice"));
  });

  test("a busy GPU or a thermal warning is not ready, with the reasons in words", async () => {
    const busy = new MetalDriver({ targetClass: "x", exec: fake(48).exec });
    assert.deepEqual((await busy.readyForTiming()).reasons, [
      "GPU 48% busy before timing (the limit is 5%)",
    ]);
    const hot = new MetalDriver({ targetClass: "x", exec: fake(0, HOT).exec });
    const r = await hot.readyForTiming();
    assert.equal(r.ready, false);
    assert.deepEqual(r.reasons, [
      "thermal warning: Thermal warning level set to 2.",
      "CPU speed limited to 70%",
    ]);
    assert.deepEqual(await hot.afterBlock(), r.reasons);
    assert.deepEqual((await hot.recover()).reasons, r.reasons);
  });

  test("the utilization limit is configurable, and an unreadable utilization is not ready", async () => {
    const d = new MetalDriver({
      targetClass: "x",
      limits: { maxIdleUtilization: 50 },
      exec: fake(48).exec,
    });
    assert.equal((await d.readyForTiming()).ready, true);
    const blind = new MetalDriver({
      targetClass: "x",
      exec: async (cmd) => (cmd === "pmset" ? CALM : ""),
    });
    assert.deepEqual((await blind.readyForTiming()).reasons, ["ioreg gave no GPU utilization"]);
    await assert.rejects(blind.identify(), /no IOPlatformUUID/);
  });

  test("records the system, the chip and the GPU's memory in use with each block", async () => {
    const d = new MetalDriver({ targetClass: "x", exec: fake(1).exec });
    assert.deepEqual(await d.envSnapshot(), {
      macos: "99.1",
      macos_build: "99A100",
      chip: "Apple M9 Test",
      gpu: "Apple M9 Test",
      gpu_cores: 40,
      utilization: 1,
      memory_in_use: 2217279488,
      warnings: [],
    });
    assert.deepEqual(await d.afterBlock(), []);
  });

  test("plausibility refuses NaN, infinite and non-positive metrics", () => {
    const d = new MetalDriver({ targetClass: "x", exec: fake(0).exec });
    assert.deepEqual(d.plausibility({ flops_per_s: 1e12 }), []);
    assert.deepEqual(d.plausibility({ a: Number.NaN, b: 0, c: Number.POSITIVE_INFINITY }), [
      "a is NaN",
      "b is 0",
      "c is Infinity",
    ]);
  });
});
