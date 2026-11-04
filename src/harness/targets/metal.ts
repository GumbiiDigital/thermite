import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { Readiness, TargetDriver, TargetIdentity } from "./driver.ts";

/**
 * The GPU of a Mac with Apple silicon, through Metal, read with the tools macOS ships: sysctl for
 * the chip, ioreg for the GPU and the machine's identity, pmset for thermal and performance
 * warnings and sw_vers for the system. It never changes a setting: it records the GPU's state
 * with every block and refuses to time when the GPU is busy or the Mac reports a thermal or
 * performance warning. Apple silicon has one GPU, shared with the display, so "idle" is judged
 * from the GPU's own utilization.
 */

const run = promisify(execFile);

/** How long one query may take before it fails (each tool answers in milliseconds). */
const QUERY_TIMEOUT_MS = 10_000;

interface MetalLimits {
  /** "Device Utilization %" above this before a block means something else is using the GPU. */
  maxIdleUtilization: number;
}

interface MetalDriverOptions {
  targetClass: string;
  limits?: Partial<MetalLimits> | undefined;
  /** For tests: replaces running the command (its program and arguments) and returns its output. */
  exec?: ((cmd: string, args: string[]) => Promise<string>) | undefined;
}

/** The IOAccelerator entry's GPU fields, as `ioreg -r -d 1 -c IOAccelerator` prints them. */
interface GpuState {
  model: string | undefined;
  cores: number | undefined;
  utilization: number | undefined;
  memoryInUse: number | undefined;
}

/** A number field `"name"=123` (inside a dictionary) or `"name" = 123` (a property). */
function numberField(text: string, name: string): number | undefined {
  const quoted = name.replace(/[.*+?^${}()|[\]\\%]/g, "\\$&");
  const m = new RegExp(`"${quoted}"\\s*=\\s*(\\d+)`).exec(text);
  return m ? Number(m[1]) : undefined;
}

function stringField(text: string, name: string): string | undefined {
  return new RegExp(`"${name}"\\s*=\\s*"([^"]*)"`).exec(text)?.[1];
}

export function parseAccelerator(ioreg: string): GpuState {
  return {
    model: stringField(ioreg, "model"),
    cores: numberField(ioreg, "gpu-core-count"),
    utilization: numberField(ioreg, "Device Utilization %"),
    memoryInUse: numberField(ioreg, "In use system memory"),
  };
}

/**
 * The warnings in `pmset -g therm`: a thermal or performance warning level other than 0, or a CPU
 * speed limit below 100%. "No ... has been recorded" means none.
 */
export function thermalWarnings(pmset: string): string[] {
  const out: string[] = [];
  for (const raw of pmset.split("\n")) {
    const line = raw.trim();
    if (!line || /\bno\b.*\bhas been recorded\b/i.test(line)) continue;
    const kind = /thermal warning/i.test(line)
      ? "thermal"
      : /performance warning/i.test(line)
        ? "performance"
        : undefined;
    if (kind) {
      const level = /(-?\d+)\s*\.?\s*$/.exec(line)?.[1];
      if (level !== "0") out.push(`${kind} warning: ${line}`);
      continue;
    }
    const limit = /CPU_Speed_Limit\s*=\s*(\d+)/.exec(line)?.[1];
    if (limit !== undefined && Number(limit) < 100) out.push(`CPU speed limited to ${limit}%`);
  }
  return out;
}

export class MetalDriver implements TargetDriver {
  readonly type = "metal-gpu";
  private readonly targetClass: string;
  private readonly limits: MetalLimits;
  private readonly exec: (cmd: string, args: string[]) => Promise<string>;

  constructor(opts: MetalDriverOptions) {
    this.targetClass = opts.targetClass;
    this.limits = { maxIdleUtilization: 5, ...opts.limits };
    this.exec =
      opts.exec ??
      (async (cmd, args) =>
        (await run(cmd, args, { encoding: "utf8", timeout: QUERY_TIMEOUT_MS })).stdout);
  }

  private async gpu(): Promise<GpuState> {
    return parseAccelerator(await this.exec("ioreg", ["-r", "-d", "1", "-c", "IOAccelerator"]));
  }

  private async therm(): Promise<string[]> {
    return thermalWarnings(await this.exec("pmset", ["-g", "therm"]));
  }

  async identify(): Promise<TargetIdentity> {
    const [chip, gpu, platform] = await Promise.all([
      this.exec("sysctl", ["-n", "machdep.cpu.brand_string"]),
      this.gpu(),
      this.exec("ioreg", ["-rd1", "-c", "IOPlatformExpertDevice"]),
    ]);
    const id = stringField(platform, "IOPlatformUUID");
    if (!id) throw new Error("ioreg gave no IOPlatformUUID: the Mac's identity is unknown");
    return {
      type: this.type,
      class: this.targetClass,
      id,
      details: { chip: chip.trim(), gpu: gpu.model, gpu_cores: gpu.cores },
    };
  }

  async readyForTiming(): Promise<Readiness> {
    const [gpu, warnings] = await Promise.all([this.gpu(), this.therm()]);
    const why: string[] = [];
    // Unreadable is not "fine": no utilization means not ready.
    if (gpu.utilization === undefined) why.push("ioreg gave no GPU utilization");
    else if (gpu.utilization > this.limits.maxIdleUtilization)
      why.push(
        `GPU ${gpu.utilization}% busy before timing (the limit is ${this.limits.maxIdleUtilization}%)`,
      );
    why.push(...warnings);
    return {
      ready: why.length === 0,
      reasons: why,
      details: {
        utilization: gpu.utilization,
        memory_in_use: gpu.memoryInUse,
        warnings,
      },
    };
  }

  async envSnapshot(): Promise<Record<string, unknown>> {
    const [version, build, chip, gpu, warnings] = await Promise.all([
      this.exec("sw_vers", ["-productVersion"]),
      this.exec("sw_vers", ["-buildVersion"]),
      this.exec("sysctl", ["-n", "machdep.cpu.brand_string"]),
      this.gpu(),
      this.therm(),
    ]);
    return {
      macos: version.trim(),
      macos_build: build.trim(),
      chip: chip.trim(),
      gpu: gpu.model,
      gpu_cores: gpu.cores,
      utilization: gpu.utilization,
      memory_in_use: gpu.memoryInUse,
      warnings,
    };
  }

  async afterBlock(): Promise<string[]> {
    return this.therm();
  }

  plausibility(metrics: Record<string, number>): string[] {
    return Object.entries(metrics)
      .filter(([, v]) => !Number.isFinite(v) || v <= 0)
      .map(([k, v]) => `${k} is ${v}`);
  }

  async recover(): Promise<Readiness> {
    return this.readyForTiming();
  }
}
