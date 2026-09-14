// Port of flybrain/mb/model.py. Verified against Python by tools/check_mb_parity.mjs.
// Weights are measured MaleCNS connectivity; the learning rule is a simplified
// model of a real mechanism. See the page text.

function rowIndex(sp, nRows) {
  const rows = Array.from({ length: nRows }, () => []);
  for (let i = 0; i < sp.v.length; i++) rows[sp.r[i]].push(i);
  return rows;
}

export class MushroomBody {
  constructor(data) {
    this.d = data;
    this.nPn = data.w_pn_kc.shape[0];
    this.nKc = data.w_pn_kc.shape[1];
    this.nMbon = data.w_kc_mbon.shape[1];
    this.nDan = data.w_dan_mbon.shape[0];

    this.pnRows = rowIndex(data.w_pn_kc, this.nPn);
    this.kcRows = rowIndex(data.w_kc_mbon, this.nKc);
    this.danRows = rowIndex(data.w_dan_mbon, this.nDan);

    this.gain = new Float64Array(data.w_kc_mbon.v.length).fill(1);
    this.valence = Float64Array.from(data.mbon_valence);
    this.punish = data.dan_is_punishment;

    this.sparsity = data.meta.sparsity;
    this.lr = data.meta.learning_rate;
    this.decay = data.meta.decay;
    this.lateral = data.meta.lateral;
    this.k = Math.max(1, Math.round(this.sparsity * this.nKc));

    this.plasticityOn = true;
    this.dopamineOn = true;
    this.kc = new Float64Array(this.nKc);
    this.mbon = new Float64Array(this.nMbon);
    this.da = new Float64Array(this.nMbon);
    // per-Kenyon-cell depression, for the 3D view
    this.kcDepression = new Float64Array(this.nKc);
  }

  reset() {
    this.gain.fill(1);
    this.kc.fill(0); this.mbon.fill(0); this.da.fill(0);
    this.kcDepression.fill(0);
  }

  /** Sparse odour code. k-winners-take-all stands in for APL inhibition. */
  kenyon(pn) {
    const drive = new Float64Array(this.nKc);
    const sp = this.d.w_pn_kc;
    for (let p = 0; p < this.nPn; p++) {
      const a = pn[p];
      if (!a) continue;
      for (const i of this.pnRows[p]) drive[sp.c[i]] += a * sp.v[i];
    }
    return this.kenyonFromDrive(drive);
  }

  /** Precompute this for a pure odour and scale it by concentration. */
  pnDrive(pn) {
    const drive = new Float64Array(this.nKc);
    const sp = this.d.w_pn_kc;
    for (let p = 0; p < this.nPn; p++) {
      const a = pn[p];
      if (!a) continue;
      for (const i of this.pnRows[p]) drive[sp.c[i]] += a * sp.v[i];
    }
    return drive;
  }

  kenyonFromDrive(drive) {
    const sorted = Array.from(drive).sort((x, y) => y - x);
    const cut = sorted[this.k - 1];
    let top = 0;
    for (let i = 0; i < this.nKc; i++) {
      this.kc[i] = drive[i] >= cut && cut > 0 ? drive[i] : 0;
      if (this.kc[i] > top) top = this.kc[i];
    }
    if (top > 0) for (let i = 0; i < this.nKc; i++) this.kc[i] /= top;
    return this.kc;
  }

  dopamineAt(punishLevel, rewardLevel) {
    this.da.fill(0);
    if (!this.dopamineOn) return this.da;
    const sp = this.d.w_dan_mbon;
    for (let dn = 0; dn < this.nDan; dn++) {
      const a = this.punish[dn] ? punishLevel : rewardLevel;
      if (!a) continue;
      for (const i of this.danRows[dn]) this.da[sp.c[i]] += a * sp.v[i];
    }
    return this.da;
  }

  /** Present a precomputed Kenyon-cell drive. The odour blend is linear in
   *  concentration, so callers precompute one drive vector per odour. */
  presentDrive(drive, punish = 0, learn = true) {
    const keep = learn ? null : Float64Array.from(this.gain);
    this.kenyonFromDrive(drive);
    const r = this.finish(punish, 0, learn);
    if (keep) this.gain = keep;
    return r.valence;
  }

  present(pn, { punish = 0, reward = 0, learn = true } = {}) {
    this.kenyon(pn);
    return this.finish(punish, reward, learn);
  }

  presentDriveLearn(drive, punish = 0, reward = 0) {
    this.kenyonFromDrive(drive);
    return this.finish(punish, reward, true).valence;
  }

  finish(punish, reward, learn) {
    const sp = this.d.w_kc_mbon;
    const raw = new Float64Array(this.nMbon);
    for (let kc = 0; kc < this.nKc; kc++) {
      const a = this.kc[kc];
      if (!a) continue;
      for (const i of this.kcRows[kc]) raw[sp.c[i]] += a * sp.v[i] * this.gain[i];
    }
    // measured MBON -> MBON interactions
    const lat = new Float64Array(this.nMbon);
    const mm = this.d.w_mbon_mbon;
    for (let i = 0; i < mm.v.length; i++) lat[mm.c[i]] += raw[mm.r[i]] * mm.v[i];
    for (let m = 0; m < this.nMbon; m++) this.mbon[m] = raw[m] + this.lateral * lat[m];

    this.dopamineAt(punish, reward);

    if (learn && this.plasticityOn) {
      if (punish || reward) {
        for (let kc = 0; kc < this.nKc; kc++) {
          const a = this.kc[kc];
          if (!a) continue;
          for (const i of this.kcRows[kc]) this.gain[i] -= this.lr * a * this.da[sp.c[i]];
        }
      }
      for (let i = 0; i < this.gain.length; i++) {
        this.gain[i] += this.decay * (1 - this.gain[i]);
        if (this.gain[i] < 0) this.gain[i] = 0;
        else if (this.gain[i] > 1) this.gain[i] = 1;
      }
    }
    this.refreshDepression();

    let v = 0;
    for (let m = 0; m < this.nMbon; m++) v += this.mbon[m] * this.valence[m];
    return { valence: v, sparsity: this.activeCount() / this.nKc };
  }

  valenceOf(pn) {
    const keep = { gain: this.gain, kc: Float64Array.from(this.kc), mbon: Float64Array.from(this.mbon) };
    this.gain = Float64Array.from(this.gain);
    const r = this.present(pn, { learn: false });
    this.gain = keep.gain;
    return r.valence;
  }

  refreshDepression() {
    const sp = this.d.w_kc_mbon;
    for (let kc = 0; kc < this.nKc; kc++) {
      const idx = this.kcRows[kc];
      if (!idx.length) { this.kcDepression[kc] = 0; continue; }
      let s = 0;
      for (const i of idx) s += this.gain[i];
      this.kcDepression[kc] = 1 - s / idx.length;
    }
  }

  activeCount() { let n = 0; for (let i = 0; i < this.nKc; i++) if (this.kc[i] > 0) n++; return n; }

  get depression() {
    let s = 0;
    for (let i = 0; i < this.gain.length; i++) s += this.gain[i];
    return 1 - s / this.gain.length;
  }

  /** Approach pool and avoidance pool totals, for the readout bars. */
  pools() {
    let approach = 0, avoid = 0;
    for (let m = 0; m < this.nMbon; m++) {
      if (this.valence[m] > 0) approach += this.mbon[m];
      else if (this.valence[m] < 0) avoid += this.mbon[m];
    }
    return { approach, avoid };
  }
}
