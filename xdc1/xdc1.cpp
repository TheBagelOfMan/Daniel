// xdc1: 4k osu!mania difficulty.
//
// build: g++ -O2 -std=c++17 xdc1.cpp -o xdc1
// usage: xdc1 [--rate r] [--graph] [chart.osu]
//
// reads a .osu file, or "column head tail" lines on stdin (tail -1 for a tap).
// define XDC1_NO_MAIN to use it as a library through xdc1::analyse().

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <istream>
#include <iterator>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xdc1 {

// shared types and grid helpers. every curve lives on a sorted time grid (ms), each value
// held from its point to the next.

using Vec = std::vector<double>;
using std::size_t;

// local pattern readings average over +-WINDOW_MS.
constexpr double WINDOW_MS = 500.0;

// a hold this short or shorter is read as a tap.
constexpr double LN_TAP_MS = 80.0;

// hit-window unit, fixed at od 9.
double hit_window_unit() {
    double x = 0.3 * std::sqrt((64.5 - std::ceil(9.0 * 3.0)) / 500.0);
    return std::min(x, 0.6 * (x - 0.09) + 0.09);
}

// index of the first element >= v (first_not_before) or > v (first_after) in sorted a.
size_t first_not_before(const Vec& a, double v) {
    return std::lower_bound(a.begin(), a.end(), v) - a.begin();
}
size_t first_after(const Vec& a, double v) {
    return std::upper_bound(a.begin(), a.end(), v) - a.begin();
}

int column_count(unsigned mask) {
    return (mask & 1) + (mask >> 1 & 1) + (mask >> 2 & 1) + (mask >> 3 & 1);
}

// integral of piecewise-constant f over [x[i] - window, x[i] + window], clipped to x.
// window_sum scales it; window_avg divides by the clipped span.
Vec window_integral(const Vec& x, const Vec& f, double window, Vec* span = nullptr) {
    const size_t n = x.size();
    Vec out(n, 0.0);
    if (span) span->assign(n, 0.0);
    if (n < 2) return out;
    Vec F(n, 0.0);
    for (size_t i = 1; i < n; ++i) F[i] = F[i - 1] + f[i - 1] * (x[i] - x[i - 1]);
    size_t ca = 0, cb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double a = std::clamp(x[i] - window, x.front(), x.back());
        const double b = std::clamp(x[i] + window, x.front(), x.back());
        while (ca < n && x[ca] < a) ++ca;
        while (cb < n && x[cb] < b) ++cb;
        const size_t ia = ca ? ca - 1 : 0, ib = cb ? cb - 1 : 0;
        out[i] = (F[ib] + f[ib] * (b - x[ib])) - (F[ia] + f[ia] * (a - x[ia]));
        if (span) (*span)[i] = b - a;
    }
    return out;
}

Vec window_sum(const Vec& x, const Vec& f, double window, double scale) {
    Vec out = window_integral(x, f, window);
    for (double& v : out) v = scale * v;
    return out;
}

Vec window_avg(const Vec& x, const Vec& f, double window) {
    Vec span;
    Vec out = window_integral(x, f, window, &span);
    for (size_t i = 0; i < out.size(); ++i) out[i] = span[i] > 0.0 ? out[i] / span[i] : 0.0;
    return out;
}

// add val held over [t0, t1) into f as each grid cell's time-average, so short spans
// count in full wherever the grid falls.
void add_span(Vec& f, const Vec& grid, double t0, double t1, double val) {
    if (grid.size() < 2 || t1 <= t0) return;
    size_t q = first_after(grid, t0);
    if (q > 0) --q;
    for (; q + 1 < grid.size() && grid[q] < t1; ++q) {
        const double lo = std::max(t0, grid[q]), hi = std::min(t1, grid[q + 1]);
        if (hi > lo) f[q] += val * (hi - lo) / (grid[q + 1] - grid[q]);
    }
}

// per-transition values on the grid: value i holds from starts[i] to starts[i + 1].
Vec steps_to_grid(const Vec& starts, const Vec& vals, const Vec& grid) {
    Vec out(grid.size(), 0.0);
    if (vals.empty()) return out;
    size_t c = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        while (c < vals.size() && starts[c] <= grid[i]) ++c;
        out[i] = vals[c ? c - 1 : 0];
    }
    return out;
}

// column, head time and tail time (ms, rate already applied). tail is -1 for a tap.
// notes are always sorted by (h, k).
struct Note {
    int k;
    int h;
    int t;
};

Vec head_times(const std::vector<Note>& notes) {
    Vec ht;
    for (const Note& n : notes) ht.push_back(n.h);
    return ht;
}

// the notes sharing one head time.
struct Row {
    double t;
    unsigned mask;
    int size;
};

std::vector<Row> build_rows(const std::vector<Note>& notes) {
    std::vector<Row> rows;
    for (const Note& n : notes) {
        if (rows.empty() || rows.back().t != n.h) rows.push_back({static_cast<double>(n.h), 0u, 0});
        rows.back().mask |= 1u << n.k;
        rows.back().size = column_count(rows.back().mask);
    }
    return rows;
}

Vec row_times(const std::vector<Row>& rows) {
    Vec times;
    for (const Row& r : rows) times.push_back(r.t);
    return times;
}

// pressing strain per hand. a hand plays with the fingers (each pressing on its own), the
// wrist (one stroke per chord or note pair), or jitter (a tensed hand vibrating through a
// motion it repeats, cheaper the faster it goes):
//
//   hand strain = (1 - w) [(1 - m) finger + m min(finger, wrist)] + w wrist,
//
// then the jittered cost where the hand jitters and it is cheaper. w (wristability) and
// m (manipulability) are read from note timing alone.

// every rate is summed over +-TECH_WINDOW_MS, per ms to per s.
constexpr double TECH_WINDOW_MS = 244.92;
constexpr double TECH_SCALE     = 0.001;

// finger: each finger's repeats plus the hand's motions, power-summed.
constexpr double MOTION_X_COEF   = 0.39104;
constexpr double MOTION_W        = 5.7561;
constexpr double MOTION_REPEAT_W = 0.55187;
constexpr double FINGER_MIX_P    = 2.3762;
// gaps read as sqrt(g^2 + FLAM_MS^2), so a near-chord costs like a chord.
constexpr double FLAM_MS         = 11.247;
constexpr double FINGER_JACK_CRIT_MS = 218.21;
// repeat penalty floor, so one fast minijack can't blow up its window.
constexpr double JACK_FLOOR_MS   = 35.809;
// a minijack faster than this is read longer, up to it, borrowing half the excess of the
// gaps either side. an even run of fast jacks has nothing to borrow.
constexpr double MINIJACK_MS     = 72.836;
// a held finger is busy until this far between head (0) and release (1).
constexpr double LN_HOLD_OCC     = 0.754;
// alternation gets dearer as its gap closes on this.
constexpr double ALT_CRIT_MS     = 21.197;

// wrist: one stroke per chord or note pair, costing WRIST_W (1 + WRIST_CHORD_W [two keys]) / g^2.
constexpr double WRIST_W         = 3.3377;
constexpr double WRIST_CHORD_W   = 0.81919;
// wrist cost grows as gap^-WRIST_GAP_POW, pivoting at WRIST_GAP_PIVOT_MS.
constexpr double WRIST_GAP_POW      = 2.6413;
constexpr double WRIST_GAP_PIVOT_MS = 100.0;
// stroke grouping: neighbours past the cap never pair; near-equal gaps pair forward.
constexpr double STROKE_CAP_MS   = 1000.0;
constexpr double STROKE_TIE_FRAC = 0.125;

// which technique. chord: 1 at phase 0 (width over the longer neighbouring gap), smoothstep
// to 0 at CHORD_PHASE. anchored: a run of ANCHOR_RUN_LO (0) to ANCHOR_RUN_HI (1) strokes
// sharing a finger.
constexpr double CHORD_PHASE       = 0.26009;
constexpr double ANCHOR_RUN_LO     = 16.0;
constexpr double ANCHOR_RUN_HI     = 48.0;
// manipulation: a pair re-timed into a chord, priced at 1 + (width / MANIP_TOL_MS)^MANIP_TOL_POW
// of the wrist stroke. only taken in a consistent pattern: the best lag (1 to
// MANIP_PERIOD_MAX notes) over +-MANIP_WINDOW notes, and the share of note pairs it misses.
constexpr double MANIP_TOL_MS      = 76.009;
constexpr double MANIP_TOL_POW     = 2.6;
constexpr int    MANIP_WINDOW      = 12;
constexpr int    MANIP_PERIOD_MAX  = 12;
constexpr double MANIP_PERIOD_COST = 0.02;
constexpr double MANIP_PACE        = 0.25;
constexpr double MANIP_PACE_FLOOR_MS = 4.0;
// notes this close are one chord to the consistency reading.
constexpr double MANIP_FLAM_MS     = 10.0;
constexpr double MANIP_CONS_LO     = 0.15;
constexpr double MANIP_CONS_HI     = 0.75;
// wristability, manipulability and jitter are averaged over +-WRIST_BLEND_MS.
constexpr double WRIST_BLEND_MS    = 64.443;

// jitter: a hand repeating one motion for JITTER_REP_LO (0) to JITTER_REP_HI (1) rows.
// under JITTER_ONSET_MS a jittered stroke costs (gap / onset)^JITTER_POW of the wrist's.
constexpr double JITTER_ONSET_MS = 95.0;
constexpr double JITTER_POW      = 1.5;
constexpr double JITTER_REP_LO   = 3.0;
constexpr double JITTER_REP_HI   = 7.0;

double soften_gap(double gap_ms) { return std::sqrt(gap_ms * gap_ms + FLAM_MS * FLAM_MS); }

// (1 / d) (1 / (d + x_term)), d in seconds.
double gap_rate(double delta_ms, double x_term) {
    const double d = 0.001 * delta_ms;
    return d > 0.0 ? (1.0 / d) * (1.0 / (d + x_term)) : 0.0;
}

Vec tech_sum(const Vec& grid, const Vec& f) { return window_sum(grid, f, TECH_WINDOW_MS, TECH_SCALE); }

double smoothstep01(double u) {
    u = std::clamp(u, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

// one hand's note: head time, finger (0 or 1), tail (-1 for a tap).
struct HandNote { double t; int f; double tail = -1.0; };

// each repeat's gap with the minijack allowance: out[i] is the gap ending at press i + 1.
Vec minijack_gaps(const Vec& t) {
    Vec g(t.size() > 1 ? t.size() - 1 : 0);
    for (size_t i = 0; i < g.size(); ++i) g[i] = t[i + 1] - t[i];
    Vec out = g;
    for (size_t i = 0; i < g.size(); ++i) {
        if (g[i] >= MINIJACK_MS) continue;
        const double before = i > 0 ? std::max(0.5 * (g[i - 1] - g[i]), 0.0) : MINIJACK_MS;
        const double after = i + 1 < g.size() ? std::max(0.5 * (g[i + 1] - g[i]), 0.0) : MINIJACK_MS;
        out[i] = std::min(g[i] + before + after, MINIJACK_MS);
    }
    return out;
}

// repeats, both fingers on one rate curve. `rice` keeps no finger busy.
Vec repeat_cost(const std::vector<HandNote>& v, const Vec& grid, bool rice) {
    std::array<Vec, 2> ft;
    for (const HandNote& n : v) ft[n.f].push_back(n.t);
    const std::array<Vec, 2> sg = {minijack_gaps(ft[0]), minijack_gaps(ft[1])};
    std::array<size_t, 2> idx = {0, 0};
    Vec rate(grid.size(), 0.0);
    std::array<double, 2> last = {-1.0, -1.0};
    std::array<double, 2> free_at = {0.0, 0.0};
    for (const HandNote& n : v) {
        double& l = last[n.f];
        if (l >= 0.0 && n.t > l) {
            const double raw = sg[n.f][idx[n.f] - 1];
            const double gap = soften_gap(raw - std::min(LN_HOLD_OCC * free_at[n.f], raw - 1.0));
            const double c = FINGER_JACK_CRIT_MS / std::sqrt(gap * gap + JACK_FLOOR_MS * JACK_FLOOR_MS);
            const double pen = c * c;
            add_span(rate, grid, l, n.t, (1.0 + pen) * gap_rate(gap, 0.0) * raw / (n.t - l));
        }
        ++idx[n.f];
        l = n.t;
        free_at[n.f] = !rice && n.tail >= n.t ? std::max(n.tail - n.t - LN_TAP_MS, 0.0) : 0.0;
    }
    return tech_sum(grid, rate);
}

// motions, each gap weighted by the motion it ends on: chord, repeat or alternation.
Vec motion_cost(double x, const std::vector<HandNote>& v, const Vec& grid) {
    Vec t;
    std::vector<unsigned> m;
    for (const HandNote& n : v) {
        if (!t.empty() && t.back() == n.t) { m.back() |= 1u << n.f; continue; }
        t.push_back(n.t);
        m.push_back(1u << n.f);
    }
    const double xt = MOTION_X_COEF * std::pow(x, 0.25);
    Vec rate(grid.size(), 0.0);
    for (size_t i = 0; i + 1 < t.size(); ++i) {
        const double gap = soften_gap(t[i + 1] - t[i]);
        const double w = m[i + 1] == 3u          ? 1.0
                         : (m[i + 1] & m[i]) != 0 ? MOTION_REPEAT_W
                                                  : 1.0 + (ALT_CRIT_MS / gap) * (ALT_CRIT_MS / gap);
        add_span(rate, grid, t[i], t[i + 1], w * gap_rate(gap, xt));
    }
    return tech_sum(grid, rate);
}

double finger_mix(double J, double S) {
    const double a = J, b = MOTION_W * S;
    const double hi = std::max(a, b);
    if (hi <= 0.0) return 0.0;
    return hi * std::pow(1.0 + std::pow(std::min(a, b) / hi, FINGER_MIX_P), 1.0 / FINGER_MIX_P);
}

// a chord, or a note plus the other finger's next press when that comes first and is not
// nearer the following press. placed at its notes' midpoint; width is how far apart they
// are. a single note's width is the distance to the other finger's nearest non-chord note.
struct Strokes {
    Vec t, width;
    std::vector<int> size;
    std::vector<unsigned> mask;   // fingers struck
    std::vector<int> lead;        // finger it starts on
    std::vector<size_t> first;    // index of its first note
};

Strokes group_strokes(const std::vector<HandNote>& v) {
    Strokes s;
    std::array<Vec, 2> ft;
    for (size_t i = 0; i < v.size(); ++i) {
        const bool chord = (i > 0 && v[i - 1].t == v[i].t) || (i + 1 < v.size() && v[i + 1].t == v[i].t);
        if (!chord) ft[v[i].f].push_back(v[i].t);
    }
    for (size_t i = 0; i < v.size();) {
        size_t j = i + 1;
        if (j < v.size() && v[j].f != v[i].f && v[j].t - v[i].t <= STROKE_CAP_MS &&
            !(j + 1 < v.size() && v[j + 1].f == v[i].f &&
              v[j + 1].t - v[j].t < (v[j].t - v[i].t) * (1.0 - STROKE_TIE_FRAC)))
            ++j;
        s.t.push_back(0.5 * (v[i].t + v[j - 1].t));
        double w = v[j - 1].t - v[i].t;
        if (j - i == 1) {
            const Vec& o = ft[1 - v[i].f];
            double d = STROKE_CAP_MS;
            const size_t q = first_not_before(o, v[i].t);
            if (q < o.size()) d = std::min(d, o[q] - v[i].t);
            if (q > 0) d = std::min(d, v[i].t - o[q - 1]);
            w = d < STROKE_CAP_MS ? d : 0.0;
        }
        s.width.push_back(w);
        s.size.push_back(static_cast<int>(j - i));
        s.lead.push_back(v[i].f);
        s.first.push_back(i);
        unsigned m = 0;
        for (size_t q = i; q < j; ++q) m |= 1u << v[q].f;
        s.mask.push_back(m);
        i = j;
    }
    return s;
}

// how well the notes around note i repeat themselves: 1 minus the misses at the best lag,
// less the lag's cost. nt/nc: both hands' note times and columns, flams read as chords.
double note_consistency(const Vec& nt, const std::vector<int>& nc, size_t i) {
    const size_t n = nt.size();
    const size_t a = i >= static_cast<size_t>(MANIP_WINDOW) ? i - MANIP_WINDOW : 0;
    const size_t b = std::min(n, i + MANIP_WINDOW + 1);
    double best = 0.0;
    for (int P = 1; P <= MANIP_PERIOD_MAX; ++P) {
        if (b < a + P + 4) break;
        double miss = 0.0;
        size_t cnt = 0;
        for (size_t q = a; q + P < b; ++q, ++cnt) {
            if (q + P + 1 >= b) break;
            if (nc[q] != nc[q + P] || nc[q + 1] != nc[q + P + 1]) { miss += 1.0; continue; }
            const double g1 = nt[q + 1] - nt[q], g2 = nt[q + P + 1] - nt[q + P];
            const double med = 0.5 * (g1 + g2);
            const double tol = std::sqrt(MANIP_PACE * med * MANIP_PACE * med + MANIP_PACE_FLOOR_MS * MANIP_PACE_FLOOR_MS);
            miss += smoothstep01(std::abs(g1 - g2) / tol);
        }
        best = std::max(best, 1.0 - miss / static_cast<double>(cnt) - MANIP_PERIOD_COST * (P - 1));
    }
    return best;
}

// each stroke's consistency: the mean of its notes', read over both hands as one sequence.
std::array<Vec, 2> stroke_consistency(const std::array<std::vector<HandNote>, 2>& hv,
                                      const std::array<Strokes, 2>& s) {
    struct Ev { double t; int c; int h; size_t q; };
    std::vector<Ev> all;
    for (int h = 0; h < 2; ++h)
        for (size_t q = 0; q < hv[h].size(); ++q) all.push_back({hv[h][q].t, 2 * h + hv[h][q].f, h, q});
    std::sort(all.begin(), all.end(), [](const Ev& a, const Ev& b) { return a.t < b.t || (a.t == b.t && a.c < b.c); });
    // notes within MANIP_FLAM_MS of a group's first share its time, columns in order.
    for (size_t e = 0; e < all.size();) {
        size_t f = e + 1;
        while (f < all.size() && all[f].t - all[e].t <= MANIP_FLAM_MS) all[f++].t = all[e].t;
        std::sort(all.begin() + e, all.begin() + f, [](const Ev& a, const Ev& b) { return a.c < b.c; });
        e = f;
    }
    Vec nt(all.size());
    std::vector<int> nc(all.size());
    for (size_t e = 0; e < all.size(); ++e) { nt[e] = all[e].t; nc[e] = all[e].c; }
    std::array<Vec, 2> per = {Vec(hv[0].size()), Vec(hv[1].size())};
    for (size_t e = 0; e < all.size(); ++e) per[all[e].h][all[e].q] = note_consistency(nt, nc, e);
    std::array<Vec, 2> out;
    for (int h = 0; h < 2; ++h) {
        out[h].resize(s[h].t.size());
        for (size_t k = 0; k < out[h].size(); ++k) {
            double sum = 0.0;
            for (int q = 0; q < s[h].size[k]; ++q) sum += per[h][s[h].first[k] + q];
            out[h][k] = sum / s[h].size[k];
        }
    }
    return out;
}

// wrist cost, the same with each pair paying its manipulation price, and the blend curves.
// jitw is how far the hand takes jitter as its own technique.
struct WristCost { Vec cost, cost_manip, able, manip, jitw; };

WristCost wrist_cost(const std::vector<HandNote>& v, const Strokes& s, const Vec& cons, const Vec& other, const Vec& grid) {
    const size_t n = s.t.size();
    // how far each stroke is one strike: a chord, or manipulable into one.
    Vec chord(n), manip(n), merged(n), strike(n);
    for (size_t k = 0; k < n; ++k) {
        double period = 0.0;
        if (k > 0) period = std::max(period, s.t[k] - s.t[k - 1]);
        if (k + 1 < n) period = std::max(period, s.t[k + 1] - s.t[k]);
        const double phase = s.width[k] <= 0.0 ? 0.0 : period > 0.0 ? s.width[k] / period : 1.0;
        const double near = 1.0 - smoothstep01(phase / CHORD_PHASE);
        const bool pair = s.size[k] == 2 && s.width[k] > 0.0;
        chord[k] = pair ? 0.0 : near;
        // a pair with an other-hand note between its notes can't be merged.
        const double lo_t = s.t[k] - 0.5 * s.width[k], hi_t = s.t[k] + 0.5 * s.width[k];
        const size_t q = first_not_before(other, lo_t);
        const bool across = q < other.size() && other[q] > lo_t && other[q] < hi_t;
        manip[k] = pair && !across ? 1.0 - (1.0 - near) * (1.0 - smoothstep01((cons[k] - MANIP_CONS_LO) / (MANIP_CONS_HI - MANIP_CONS_LO))) : 0.0;
        merged[k] = 1.0 - (1.0 - chord[k]) * (1.0 - manip[k]);
        strike[k] = s.size[k] == 1 ? 1.0 : smoothstep01((merged[k] - 0.2) / 0.6);
    }
    // anchor run: strikes in a row one finger is in throughout, soft so rates move it smoothly.
    Vec run(n, 0.0);
    for (unsigned f = 1; f <= 2; f <<= 1) {
        Vec fw(n, 0.0), bw(n, 0.0);
        for (size_t k = 0; k < n; ++k)
            if (s.mask[k] & f) fw[k] = strike[k] * (1.0 + (k > 0 && (s.mask[k - 1] & f) ? fw[k - 1] : 0.0));
        for (size_t k = n; k-- > 0;)
            if (s.mask[k] & f) bw[k] = strike[k] * (1.0 + (k + 1 < n && (s.mask[k + 1] & f) ? bw[k + 1] : 0.0));
        for (size_t k = 0; k < n; ++k)
            if (s.mask[k] & f) run[k] = std::max(run[k], fw[k] + bw[k] - strike[k]);
    }
    // repeat run for jitter: on the hand's rows and on its strokes, the longer. a gap only
    // breaks it as a real rest, 2 to 3 times the gap beside it.
    Vec rep(n, 1.0);
    {
        Vec rt;
        std::vector<unsigned> rm;
        std::vector<size_t> row_of(v.size());
        for (size_t q = 0; q < v.size(); ++q) {
            if (rt.empty() || v[q].t != rt.back()) { rt.push_back(v[q].t); rm.push_back(0u); }
            rm.back() |= 1u << v[q].f;
            row_of[q] = rt.size() - 1;
        }
        const size_t nr = rt.size();
        std::vector<char> jacked(nr, 0);
        for (size_t k = 0; k < nr; ++k)
            jacked[k] = (k > 0 && rm[k - 1] == rm[k]) || (k + 1 < nr && rm[k + 1] == rm[k]);
        auto pace = [](double g1, double g2) {
            return 1.0 - smoothstep01(std::max(g1, g2) / std::max(std::min(g1, g2), 1.0) - 2.0); };
        Vec same(nr, 0.0);
        for (size_t k = 1; k < nr; ++k)
            if (rm[k] == rm[k - 1] || (jacked[k] && jacked[k - 1] && rm[k] != 3u && rm[k - 1] != 3u))
                same[k] = pace(k >= 2 ? rt[k - 1] - rt[k - 2] : rt[k] - rt[k - 1], rt[k] - rt[k - 1]);
        Vec fw(nr, 0.0), bw(nr, 0.0);
        for (size_t k = 1; k < nr; ++k) fw[k] = same[k] * (1.0 + fw[k - 1]);
        for (size_t k = nr; k-- > 1;) bw[k - 1] = same[k] * (1.0 + bw[k]);
        Vec sm(n, 0.0), sf(n, 0.0), sb(n, 0.0);
        for (size_t k = 1; k < n; ++k)
            if (s.mask[k] == s.mask[k - 1]) sm[k] = pace(k >= 2 ? s.t[k - 1] - s.t[k - 2] : s.t[k] - s.t[k - 1], s.t[k] - s.t[k - 1]);
        for (size_t k = 1; k < n; ++k) sf[k] = sm[k] * (1.0 + sf[k - 1]);
        for (size_t k = n; k-- > 1;) sb[k - 1] = sm[k] * (1.0 + sb[k]);
        for (size_t k = 0; k < n; ++k) {
            const size_t r = row_of[s.first[k]];
            rep[k] = 1.0 + std::max(fw[r] + bw[r], sf[k] + sb[k]);
        }
    }
    Vec rate(grid.size(), 0.0), rate_m(grid.size(), 0.0), able(grid.size(), 0.0), mcurve(grid.size(), 0.0), jitw(grid.size(), 0.0);
    for (size_t i = 0; i + 1 < n; ++i) {
        const double gap = s.t[i + 1] - s.t[i];
        const double r = (s.mask[i + 1] & s.mask[i]) != 0
                             ? smoothstep01((run[i] - ANCHOR_RUN_LO) / (ANCHOR_RUN_HI - ANCHOR_RUN_LO)) : 0.0;
        const double a = 1.0 - (1.0 - chord[i]) * (1.0 - r);
        add_span(able, grid, s.t[i], s.t[i + 1], a);
        add_span(mcurve, grid, s.t[i], s.t[i + 1], manip[i]);
        const double j = smoothstep01((rep[i] - JITTER_REP_LO) / (JITTER_REP_HI - JITTER_REP_LO));
        const double share = gap > 0.0 && gap < JITTER_ONSET_MS ? std::pow(gap / JITTER_ONSET_MS, JITTER_POW) : 1.0;
        add_span(jitw, grid, s.t[i], s.t[i + 1], j * std::min(1.0, (1.0 - share) / 0.1));
        const double load = 1.0 + WRIST_CHORD_W * (s.size[i + 1] - 1.0);
        const double off = s.size[i + 1] == 2 ? s.width[i + 1] / MANIP_TOL_MS : 0.0;
        const double c = load * gap_rate(gap, 0.0) *
                         (gap > 0.0 ? std::pow(WRIST_GAP_PIVOT_MS / gap, WRIST_GAP_POW - 2.0) : 1.0) *
                         (1.0 - j * (1.0 - share));
        add_span(rate, grid, s.t[i], s.t[i + 1], c);
        add_span(rate_m, grid, s.t[i], s.t[i + 1], c * (1.0 + std::pow(off, MANIP_TOL_POW)));
    }
    WristCost o;
    o.cost = tech_sum(grid, rate);
    o.cost_manip = tech_sum(grid, rate_m);
    for (double& c : o.cost) c *= WRIST_W;
    for (double& c : o.cost_manip) c *= WRIST_W;
    o.able = window_avg(grid, able, WRIST_BLEND_MS);
    o.manip = window_avg(grid, mcurve, WRIST_BLEND_MS);
    o.jitw = window_avg(grid, jitw, WRIST_BLEND_MS);
    return o;
}

// the hand's cost at grid point i from its finger cost f: the blend, then jitter.
double hand_cost(double f, const WristCost& W, size_t i) {
    const double w = std::clamp(W.able[i], 0.0, 1.0), m = std::clamp(W.manip[i], 0.0, 1.0);
    const double h = (1.0 - w) * ((1.0 - m) * f + m * std::min(f, W.cost_manip[i])) + w * W.cost[i];
    const double j = std::clamp(W.jitw[i], 0.0, 1.0);
    return (1.0 - j) * h + j * std::min(h, W.cost_manip[i]);
}

// columns 0-1 are hand 0, 2-3 hand 1. returns each hand's strain, and fills `able` with
// each hand's wristability and `rice` with each hand's strain with every note read as a tap.
std::array<Vec, 2> technique_strain(double x, const std::array<std::vector<Note>, 4>& by_col,
                                    const Vec& grid, std::array<Vec, 2>& able,
                                    std::array<Vec, 2>& rice) {
    std::array<std::vector<HandNote>, 2> hv;
    std::array<Strokes, 2> strokes;
    for (int h = 0; h < 2; ++h) {
        std::vector<HandNote>& v = hv[h];
        for (int c = 2 * h; c < 2 * h + 2; ++c)
            for (const Note& n : by_col[c]) v.push_back({static_cast<double>(n.h), c - 2 * h, static_cast<double>(n.t)});
        std::sort(v.begin(), v.end(), [](const HandNote& a, const HandNote& b) {
            return a.t < b.t || (a.t == b.t && a.f < b.f); });
        strokes[h] = group_strokes(v);
    }
    const std::array<Vec, 2> cons = stroke_consistency(hv, strokes);

    std::array<Vec, 2> out;
    for (int h = 0; h < 2; ++h) {
        const std::vector<HandNote>& v = hv[h];
        const Vec J = repeat_cost(v, grid, false);
        const Vec S = motion_cost(x, v, grid);
        Vec other;
        for (const HandNote& n : hv[1 - h]) other.push_back(n.t);
        const WristCost W = wrist_cost(v, strokes[h], cons[h], other, grid);
        // tails only reach the repeat part, so rice only re-reads that.
        const Vec Jr = repeat_cost(v, grid, true);
        out[h].resize(grid.size());
        rice[h].resize(grid.size());
        able[h].resize(grid.size());
        for (size_t i = 0; i < grid.size(); ++i) {
            out[h][i] = hand_cost(finger_mix(J[i], S[i]), W, i);
            rice[h][i] = hand_cost(finger_mix(Jr[i], S[i]), W, i);
            // wristability for the hand combine: wrist outright, or manipulable.
            const double w = std::clamp(W.able[i], 0.0, 1.0), m = std::clamp(W.manip[i], 0.0, 1.0);
            able[h][i] = w + (1.0 - w) * m;
        }
    }
    return out;
}

// stamina per hand: an energy tank drained above the map's own level and refilled below
// it, plus slow durability loss. comes out as a strain multiplier, 1 when rested.

constexpr double STAMINA_GAIN      = 2.2908;
// level a hand is measured against, as a share of the map's own level.
constexpr double STAMINA_PROP      = 0.90996;
// an empty tank multiplies strain by this (before the gain).
constexpr double STAMINA_CAP       = 1.6526;
// tank size: play at 1.5x the level empties it in STAMINA_TANK seconds. refills with a
// time constant of REFILL_REST_MS + REFILL_LEVEL_MS rel^2.
constexpr double STAMINA_TANK      = 50.918;
constexpr double STAMINA_REFILL_REST_MS  = 5000.0;
constexpr double STAMINA_REFILL_LEVEL_MS = 15000.0;
constexpr double STAM_INTERVAL_MS = 500.0;
// durability: 1 + STAMINA_DURABILITY sqrt(intensity-weighted minutes played).
constexpr double STAMINA_DURABILITY = 0.029925;
// map level: gain (stretches whose +-STAMINA_EASY_WINDOW_MS average is under STAMINA_EASY
// of the hard part's level) is left out. the hard part's level is a lehmer mean weighted
// by strain^STAMINA_LEVEL_POW, which easy stretches can't move.
constexpr double STAMINA_LEVEL_POW = 5.0;
constexpr double STAMINA_EASY = 0.17545;
constexpr double STAMINA_EASY_WINDOW_MS = 4000;

// weighted 5-power mean of both hands pooled, over the map's playing only (gain left
// out). one level for both hands.
double map_level(const Vec& grid, const std::array<Vec, 2>& strain,
                 const std::array<Vec, 2>& w) {
    const size_t n = strain[0].size();
    Vec both(n);
    for (size_t i = 0; i < n; ++i) both[i] = 0.5 * (strain[0][i] + strain[1][i]);
    const Vec g = window_avg(grid, both, STAMINA_EASY_WINDOW_MS);
    double acc = 0.0, tw = 0.0;
    for (int h = 0; h < 2; ++h)
        for (size_t i = 0; i < n; ++i) {
            if (!(g[i] > 0.0)) continue;
            const double wd = std::pow(g[i], STAMINA_LEVEL_POW) * w[h][i];
            acc += wd * g[i];
            tw += wd;
        }
    if (tw <= 0.0 || acc <= 0.0) return 0.0;
    const double easy = STAMINA_EASY * acc / tw;
    double acc5 = 0.0, tw5 = 0.0;
    for (int h = 0; h < 2; ++h)
        for (size_t i = 0; i < n; ++i) {
            if (g[i] < easy) continue;
            const double d = strain[h][i], d2 = d * d;
            acc5 += d2 * d2 * d * w[h][i];
            tw5 += w[h][i];
        }
    if (tw5 <= 0.0 || acc5 <= 0.0) return 0.0;
    return std::pow(acc5 / tw5, 0.2);
}

Vec run_tank(const Vec& grid, const Vec& strain, double level) {
    Vec mod_curve(strain.size(), 1.0);
    double tank = 1.0;
    double prev = 0.0;
    double work = 0.0;
    for (size_t i = 0; i < strain.size(); ++i) {
        const double step = i == 0 ? 0.0 : (grid[i] - grid[i - 1]) / STAM_INTERVAL_MS;
        const double avs = 0.5 * (prev + strain[i]);
        prev = strain[i];

        const double rel = avs / level;
        if (rel > 1.0) {
            tank -= (rel - 1.0) * step / STAMINA_TANK;
        } else {
            const double tau = STAMINA_REFILL_REST_MS + STAMINA_REFILL_LEVEL_MS * rel * rel;
            tank += (1.0 - tank) * (1.0 - std::exp(-step * STAM_INTERVAL_MS / tau));
        }
        tank = std::max(tank, 0.0);
        work += rel * step * STAM_INTERVAL_MS / 60000.0;
        mod_curve[i] = (1.0 + (STAMINA_CAP - 1.0) * (1.0 - tank)) *
                       (1.0 + STAMINA_DURABILITY * std::sqrt(work));
    }
    return mod_curve;
}

// w: each hand's note weights. strain is left as is when there is nothing to measure against.
void apply_stamina(const Vec& grid, std::array<Vec, 2>& strain, const std::array<Vec, 2>& w) {
    const double level = STAMINA_PROP * map_level(grid, strain, w);
    if (!(level > 0.0)) return;
    for (int h = 0; h < 2; ++h) {
        const Vec mod = run_tank(grid, strain[h], level);
        for (size_t i = 0; i < mod.size(); ++i)
            strain[h][i] *= std::max(0.0, 1.0 + STAMINA_GAIN * (mod[i] - 1.0));
    }
}

// both hands' strain into one curve.

// strain = (left^p + right^p)^(1 / p), p sliding from HAND_P_ALT (loads add) to
// HAND_P_SYNC (nearly the harder hand alone) with sync.
constexpr double HAND_P_SYNC = 6.1324;
constexpr double HAND_P_ALT  = 2.526;
// sync = (1 - SWITCH_MIX) wristability + SWITCH_MIX (1 - hand switch share).
constexpr double SWITCH_MIX = 0.16864;
constexpr double SWITCH_WINDOW_MS = 500.0;

// how much the hands work as one unit, 0..1: mean wristability mixed with how rarely
// consecutive single notes switch hands.
Vec hand_sync(const std::array<Vec, 2>& able, const std::vector<Row>& rows, const Vec& grid) {
    Vec s(able[0].size(), 0.0);
    Vec sw(grid.size(), 0.0);
    // one entry per consecutive pair of single-note rows: 1 if it changed hands.
    Vec pt, ch;
    for (size_t r = 1; r < rows.size(); ++r) {
        const unsigned a = rows[r - 1].mask, b = rows[r].mask;
        if (rows[r - 1].size != 1 || rows[r].size != 1) continue;
        pt.push_back(rows[r].t);
        ch.push_back(((a & 3) != 0) != ((b & 3) != 0) ? 1.0 : 0.0);
    }
    Vec cum(ch.size() + 1, 0.0);
    for (size_t k = 0; k < ch.size(); ++k) cum[k + 1] = cum[k] + ch[k];
    size_t lo = 0, hi = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
        while (lo < pt.size() && pt[lo] < grid[i] - SWITCH_WINDOW_MS) ++lo;
        while (hi < pt.size() && pt[hi] <= grid[i] + SWITCH_WINDOW_MS) ++hi;
        sw[i] = hi > lo ? (cum[hi] - cum[lo]) / static_cast<double>(hi - lo) : 0.0;
    }
    for (size_t i = 0; i < s.size(); ++i) {
        const double a = std::clamp(0.5 * (able[0][i] + able[1][i]), 0.0, 1.0);
        s[i] = std::clamp((1.0 - SWITCH_MIX) * a + SWITCH_MIX * (1.0 - sw[i]), 0.0, 1.0);
    }
    return s;
}

// strain = (left^p + right^p)^(1 / p), p from sync.
Vec combine_hands(const std::array<Vec, 2>& strain, const Vec& sync) {
    Vec out(strain[0].size(), 0.0);
    for (size_t i = 0; i < out.size(); ++i) {
        const double a = std::max(strain[0][i], 0.0), b = std::max(strain[1][i], 0.0);
        const double hi = std::max(a, b);
        if (hi <= 0.0) continue;
        if (std::min(a, b) <= 0.0) { out[i] = hi; continue; }
        const double s = std::clamp(sync[i], 0.0, 1.0);
        const double p = HAND_P_ALT + (HAND_P_SYNC - HAND_P_ALT) * s;
        const double lo = std::min(a, b) / hi;
        out[i] = hi * std::pow(1.0 + std::pow(lo, p), 1.0 / p);
    }
    return out;
}

// long note cost beyond a tap: release timing, a flat weight per hold, and pressing
// while another column is held.

// each factor summed over +-WINDOW_MS. the *_held curves count only holds longer than
// LN_TAP_MS and feed the release skillset.
struct HoldCosts { Vec rel, weight, cross, weight_held, cross_held; };

// w[held column][pressed column]: 1.00 within a hand, 0.20 across, 0.15 outermost.
constexpr std::array<std::array<double, 4>, 4> HOLD_PRESS_W = {{
    {0.00, 1.00, 0.20, 0.15},
    {1.00, 0.00, 0.20, 0.20},
    {0.20, 0.20, 0.00, 1.00},
    {0.15, 0.20, 1.00, 0.00},
}};

constexpr double LN_WEIGHT      = 0.05;
constexpr double REL_GAIN       = 0.75;
constexpr double REL_MID        = 0.75;
constexpr double REL_GAP_WEIGHT = 0.5;
constexpr double LX_GAIN        = 0.8;
constexpr double LX_RAMP_MS     = 150.0;
constexpr double LX_STACK       = 1.0;

// point events are placed as 1000x their value held for 1 ms, so after the 0.001-scaled
// window sum each is worth its value.
// by_col: this hand's columns only; note_seq: this hand's notes.
HoldCosts hold_costs(double x, const std::array<std::vector<Note>, 4>& by_col,
                     const std::vector<Note>& note_seq, const Vec& grid) {
    HoldCosts out;
    std::vector<Note> tail_seq;
    for (const Note& n : note_seq) if (n.t >= 0) tail_seq.push_back(n);
    std::stable_sort(tail_seq.begin(), tail_seq.end(), [](const Note& p, const Note& q) { return p.t < q.t; });
    Vec R_step(grid.size(), 0.0), W_step(grid.size(), 0.0), X_step(grid.size(), 0.0);
    Vec Wh_step(grid.size(), 0.0), Xh_step(grid.size(), 0.0);
    const double inv_x = 1.0 / x;
    auto sigmoid = [](double I) { return 1.0 / (1.0 + std::exp(-5.0 * (I - REL_MID))); };

    std::array<Vec, 4> col_times;
    for (int k = 0; k < 4; ++k) col_times[k] = head_times(by_col[k]);

    // release: scored by hold length past LN_TAP_MS and the gap to the next head in the
    // column, paced by consecutive tails.
    std::vector<const Note*> rel;
    Vec I_list;
    for (const Note& n : tail_seq) {
        const double len = static_cast<double>(n.t - n.h);
        if (len <= LN_TAP_MS) continue;
        double h_j = 1e9;
        const Vec& times = col_times[n.k];
        size_t idx = first_not_before(times, static_cast<double>(n.h));
        if (idx + 1 < times.size()) h_j = times[idx + 1];
        const double s_h = sigmoid(0.001 * (len - LN_TAP_MS) * inv_x);
        const double s_t = sigmoid(0.001 * std::max(h_j - n.t, 0.0) * inv_x);
        rel.push_back(&n);
        I_list.push_back((s_h + REL_GAP_WEIGHT * s_t) / (1.0 + REL_GAP_WEIGHT));
    }
    for (size_t i = 0; i + 1 < rel.size(); ++i) {
        double t_start = rel[i]->t, t_end = rel[i + 1]->t;
        if (t_end <= t_start) continue;
        double delta_r = 0.001 * (t_end - t_start);
        double val = REL_GAIN * 0.08 / std::sqrt(delta_r) * inv_x *
                     (1.0 + 0.8 * (I_list[i] + I_list[i + 1]));
        add_span(R_step, grid, t_start, t_end, val);
    }

    // weight: flat per hold, on its head.
    for (const Note& n : note_seq) {
        if (n.t < 0) continue;
        const double th = static_cast<double>(n.h);
        add_span(W_step, grid, th, th + 1.0, 1000.0 * LN_WEIGHT);
        if (n.t - n.h > LN_TAP_MS) add_span(Wh_step, grid, th, th + 1.0, 1000.0 * LN_WEIGHT);
    }

    // cross: each press against every hold live in another column, ramping in over
    // LX_RAMP_MS. presses on a hold's head or tail are free.
    std::array<std::vector<const Note*>, 4> holds;
    for (int k = 0; k < 4; ++k)
        for (const Note& n : by_col[k]) if (n.t >= 0) holds[k].push_back(&n);
    std::array<size_t, 4> cur{};

    for (const Note& press : note_seq) {
        const double t = static_cast<double>(press.h);
        double sum = 0.0, worst = 0.0, sum_h = 0.0, worst_h = 0.0;
        for (int k = 0; k < 4; ++k) {
            if (k == press.k) continue;
            const std::vector<const Note*>& hk = holds[k];
            while (cur[k] < hk.size() && static_cast<double>(hk[cur[k]]->t) < t) ++cur[k];
            if (cur[k] >= hk.size()) continue;
            const Note& h = *hk[cur[k]];
            double d = std::min(t - static_cast<double>(h.h), static_cast<double>(h.t) - t);
            if (d <= 0.0) continue;
            const double pair = HOLD_PRESS_W[k][press.k] * std::min(1.0, d / LX_RAMP_MS);
            sum += pair;
            worst = std::max(worst, pair);
            if (h.t - h.h > LN_TAP_MS) { sum_h += pair; worst_h = std::max(worst_h, pair); }
        }
        if (sum <= 0.0) continue;
        const double acc = worst + LX_STACK * (sum - worst);
        add_span(X_step, grid, t, t + 1.0, 1000.0 * LX_GAIN * acc);
        const double acc_h = worst_h + LX_STACK * (sum_h - worst_h);
        add_span(Xh_step, grid, t, t + 1.0, 1000.0 * LX_GAIN * acc_h);
    }

    out.rel = window_sum(grid, R_step, WINDOW_MS, 0.001);
    out.weight = window_sum(grid, W_step, WINDOW_MS, 0.001);
    out.cross = window_sum(grid, X_step, WINDOW_MS, 0.001);
    out.weight_held = window_sum(grid, Wh_step, WINDOW_MS, 0.001);
    out.cross_held = window_sum(grid, Xh_step, WINDOW_MS, 0.001);
    return out;
}

// expected-accuracy aggregation: a map's rating is the skill that scores 98% over the
// whole strain curve, then note-count damping and the output scale.

constexpr double RATING_SCALE = 7.528655953984781;

// each grid point of strain d costs a player of skill s an expected 5% (d / s)^AGG_POWER
// of its accuracy, bending toward all of it. the rating is the skill that loses
// AGG_TARGET_LOSS over the map: the skill that scores 98%.
constexpr double AGG_POWER = 3.8724;
constexpr double AGG_TARGET_LOSS = 0.02;
constexpr double SKILLSET_SHARE_POW = 0.7;
constexpr int SOLVE_STEPS = 40;
// note-count damping: raw rating times n / (n + NOTE_DAMP_N).
constexpr double NOTE_DAMP_N = 60.0;

// display scale: points = SCALE_A * ln(1 + raw / SCALE_C).
constexpr double SCALE_A = 30.0;
constexpr double SCALE_C = 290.0;

// the curve log-binned once (1% wide).
struct BinnedStrain {
    std::vector<int> bin;   // per grid point, -1 for zero strain
    Vec dp;                 // per bin, mean strain ^ AGG_POWER
};

BinnedStrain bin_strain(const Vec& strain) {
    BinnedStrain p;
    p.bin.assign(strain.size(), -1);
    const double lo = std::log(1e-4), step = std::log(1.01);
    std::map<int, std::vector<size_t>> groups;
    for (size_t i = 0; i < strain.size(); ++i)
        if (strain[i] > 1e-4) groups[static_cast<int>((std::log(strain[i]) - lo) / step)].push_back(i);
    for (const auto& [key, idx] : groups) {
        double sum = 0.0;
        for (size_t i : idx) sum += strain[i];
        for (size_t i : idx) p.bin[i] = static_cast<int>(p.dp.size());
        p.dp.push_back(std::pow(sum / static_cast<double>(idx.size()), AGG_POWER));
    }
    return p;
}

// bisects on log s for the skill whose expected loss is AGG_TARGET_LOSS of mean_w.
double solve_skill(const Vec& bw, const Vec& dp, double mean_w) {
    const double target = AGG_TARGET_LOSS * mean_w;
    double cap = 0.0, dmax = 0.0;
    for (size_t b = 0; b < bw.size(); ++b)
        if (bw[b] > 0.0) { cap += bw[b]; dmax = std::max(dmax, dp[b]); }
    if (cap <= target || dmax <= 0.0) return 0.0;
    double lo = std::log(dmax) - 200.0, hi = std::log(dmax) + 60.0;
    for (int it = 0; it < SOLVE_STEPS; ++it) {
        const double mid = 0.5 * (lo + hi), inv = std::exp(-mid);
        double loss = 0.0;
        for (size_t b = 0; b < bw.size(); ++b)
            if (bw[b] > 0.0) loss += bw[b] * (1.0 - std::exp(-0.05 * dp[b] * inv));
        if (loss > target) lo = mid; else hi = mid;
    }
    return std::exp(0.5 * (lo + hi) / AGG_POWER);
}

// norm_w > 0 takes the loss budget over that weight instead of w's own total.
double rating_from_bins(const BinnedStrain& pd, const Vec& w, double total_notes,
                        double norm_w = 0.0) {
    const double total_w = std::accumulate(w.begin(), w.end(), 0.0);
    if (total_w <= 0.0) return 0.0;
    Vec bw(pd.dp.size(), 0.0);
    for (size_t i = 0; i < w.size(); ++i)
        if (pd.bin[i] >= 0) bw[pd.bin[i]] += w[i];
    double raw = solve_skill(bw, pd.dp, norm_w > 0.0 ? norm_w : total_w);
    raw *= total_notes / (total_notes + NOTE_DAMP_N);
    return raw * 0.975 * RATING_SCALE;
}

// note count in +-WINDOW_MS around each grid point times the time it stands for.
Vec note_weights(const Vec& grid, const std::vector<Note>& notes) {
    const Vec ht = head_times(notes);
    const size_t n = grid.size();
    Vec w(n, 0.0);
    size_t lo = 0, hi = 0;
    for (size_t i = 0; i < n; ++i) {
        while (lo < ht.size() && ht[lo] < grid[i] - WINDOW_MS) ++lo;
        while (hi < ht.size() && ht[hi] < grid[i] + WINDOW_MS) ++hi;
        const double gap = i == 0     ? (grid[1] - grid[0]) / 2.0
                         : i == n - 1 ? (grid[n - 1] - grid[n - 2]) / 2.0
                                      : (grid[i + 1] - grid[i - 1]) / 2.0;
        w[i] = static_cast<double>(hi - lo) * gap;
    }
    return w;
}

// w_skill: the map's weights times the skillset's membership. capped at overall.
double skillset_rating(const BinnedStrain& pd, const Vec& w_skill, double total_w,
                       double total_notes, double overall) {
    const double sum_w = std::accumulate(w_skill.begin(), w_skill.end(), 0.0);
    const double share = total_w > 0.0 ? sum_w / total_w : 0.0;
    // note-count credit: slope 1 at 0, saturating to the full count.
    const double credit = std::min(share + 2.0 * share * share * (1.0 - share), 1.0);
    const double norm = std::pow(sum_w, SKILLSET_SHARE_POW) *
                        std::pow(total_w, 1.0 - SKILLSET_SHARE_POW);
    return std::min(rating_from_bins(pd, w_skill, total_notes * credit, norm), overall);
}

// raw rating to the displayed number.
double rating_points(double raw) {
    return raw > 0.0 ? SCALE_A * std::log1p(raw / SCALE_C) : 0.0;
}

// a strain value on the same scale, for graphs.
double strain_points(double strain) { return rating_points(strain * 0.975 * RATING_SCALE); }

// pattern classifier: seven 0..1 membership curves. each transition goes through the
// stream gateway (stream / jumpstream / handstream) or the jack gateway (speedjack /
// chordjack), with technical alongside and release from the hold costs.

constexpr int NUM_SKILLSETS = 7;
constexpr std::array<const char*, NUM_SKILLSETS> SKILLSETS = {
    "stream", "jumpstream", "handstream", "speedjack", "chordjack", "technical", "release"};
enum SkillIdx { SS_STREAM = 0, SS_JUMPSTREAM, SS_HANDSTREAM, SS_SPEEDJACK,
                SS_CHORDJACK, SS_TECHNICAL, SS_RELEASE };
constexpr int NUM_PATTERN_SKILLSETS = 6;
using Memberships = std::array<Vec, NUM_PATTERN_SKILLSETS>;

constexpr double PI = 3.14159265358979323846;

// gateways. the window tapers over its last WINDOW_TAPER_MS so row counts are
// continuous in rate.
constexpr double WINDOW_TAPER_MS = 50.0;
constexpr double STREAM_JS_DENSITY = 1.25;   // 2s1s1s1
constexpr double JS_SPLIT_SLOPE = 60.0;
constexpr int HANDSTREAM_REACH = 2;
constexpr double JACK_ANCHOR_MID = 4.5;
constexpr double JACK_DENSITY_MID = 2.3;
constexpr double JACK_ANCHOR_SLOPE = 1.0;
constexpr double JACK_DENSITY_SLOPE = 1.5;
constexpr double JACK_ANCHOR_W = 0.46;       // density takes the rest
constexpr double JACK_BINARY_SHARP = 6.0;
constexpr double ANCHOR_FULL_RUNS = 0.5;
constexpr int STREAM_BRACKET_MAX = 1;

// graces.
constexpr double GRACE_RATIO = 0.30;
constexpr int GRACE_REACH = 3;
constexpr double GRACE_FULL_MS = 30.0;
constexpr double GRACE_MAX_MS = 60.0;
constexpr double GRACE_GROUP_MAX_MS = 70.0;

// technical.
constexpr double TECH_CHAOS_W = 1.50;
constexpr double TECH_MINIJACK_W = 2.00;
constexpr double TECH_BALANCE_W = 2.00;
constexpr int TECH_MINIJACK_MAX_RUN = 2;
constexpr double TECH_MINIJACK_FLOOR = 0.45;
constexpr double TECH_MINIJACK_JACK_LO = 0.20;
constexpr double TECH_MINIJACK_JACK_HI = 0.45;
constexpr double TECH_CHAOS_DEADBAND = 0.15;
constexpr double TECH_BALANCE_DEADBAND = 0.10;
constexpr double TECH_GAIN = 4.0;
constexpr double TECH_SPEED_FLOOR_MS = 150.0;
constexpr double TECH_BREAK_RATIO = 4.0;
constexpr double TECH_RES_GAIN = 0.5;
constexpr size_t TECH_BALANCE_ROWS = 4;

// release. the flat hold weight is heard only lightly, so tap-length holds don't read
// as release.
constexpr double RELEASE_TAIL_MS = 150.0;
constexpr double RELEASE_MEMB_WEIGHT = 0.05;

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// push a 0..1 blend away from 0.5.
double sharpen_split(double b) {
    return 0.5 + 0.5 * std::tanh(JACK_BINARY_SHARP * (b - 0.5)) / std::tanh(0.5 * JACK_BINARY_SHARP);
}

bool transition_is_jack(const Row& a, const Row& b) { return (a.mask & b.mask) != 0u; }

// shared columns beyond the a+b-4 that four columns force.
int unforced_overlap(const Row& a, const Row& b) {
    return column_count(a.mask & b.mask) - std::max(0, a.size + b.size - 4);
}

// unforced overlap, or a jump against a hand.
bool minijack_pair(const Row& a, const Row& b) {
    if (unforced_overlap(a, b) > 0) return true;
    return std::min(a.size, b.size) == 2 && std::max(a.size, b.size) == 3;
}

std::vector<char> jack_flags(const std::vector<Row>& rows) {
    std::vector<char> f;
    if (rows.size() < 2) return f;
    f.reserve(rows.size() - 1);
    for (size_t i = 0; i + 1 < rows.size(); ++i)
        f.push_back(transition_is_jack(rows[i], rows[i + 1]) ? 1 : 0);
    return f;
}

// 1 inside the window, ramping to 0 over the last WINDOW_TAPER_MS.
double window_weight(double dt) {
    const double slack = WINDOW_MS - std::fabs(dt);
    if (slack >= WINDOW_TAPER_MS) return 1.0;
    return slack > 0.0 ? slack / WINDOW_TAPER_MS : 0.0;
}

template <typename Weight>
double window_mean(const std::vector<Row>& rows, const Vec& times, double center, Weight weight) {
    const size_t lo = first_not_before(times, center - WINDOW_MS);
    const size_t hi = first_after(times, center + WINDOW_MS);
    if (hi <= lo) return 1.0;
    double s = 0.0, n = 0.0;
    for (size_t i = lo; i < hi; ++i) {
        const double w = window_weight(times[i] - center);
        s += w * weight(rows[i]);
        n += w;
    }
    return (n > 0.0) ? s / n : 1.0;
}

// average row size.
double window_density(const std::vector<Row>& rows, const Vec& times, double center) {
    return window_mean(rows, times, center, [](const Row& r) { return static_cast<double>(r.size); });
}

// jump content: only a jump counts as 2.
double window_jump_density(const std::vector<Row>& rows, const Vec& times, double center) {
    return window_mean(rows, times, center, [](const Row& r) { return r.size == 2 ? 2.0 : 1.0; });
}

// a transition is handstream when a hand sits within HANDSTREAM_REACH rows of it.
std::vector<char> handstream_flags(const std::vector<Row>& rows) {
    size_t n = rows.size();
    std::vector<char> hs(n < 2 ? 0 : n - 1, 0);
    if (n < 2) return hs;
    const int FAR = static_cast<int>(n) + 1;
    std::vector<int> dist(n, FAR);
    int last = -FAR;
    for (size_t i = 0; i < n; ++i) {
        if (rows[i].size >= 3) last = static_cast<int>(i);
        dist[i] = static_cast<int>(i) - last;
    }
    last = 2 * FAR;
    for (size_t i = n; i-- > 0;) {
        if (rows[i].size >= 3) last = static_cast<int>(i);
        dist[i] = std::min(dist[i], last - static_cast<int>(i));
    }
    for (size_t i = 0; i + 1 < n; ++i)
        hs[i] = std::min(dist[i], dist[i + 1]) <= HANDSTREAM_REACH ? 1 : 0;
    return hs;
}

// longest same-column run each row belongs to.
std::vector<int> anchor_run_lengths(const std::vector<Row>& rows) {
    std::vector<int> best(rows.size(), 1);
    for (int c = 0; c < 4; ++c) {
        size_t i = 0;
        while (i < rows.size()) {
            if (!(rows[i].mask & (1u << c))) { ++i; continue; }
            size_t j = i;
            while (j < rows.size() && (rows[j].mask & (1u << c))) ++j;
            int len = static_cast<int>(j - i);
            for (size_t q = i; q < j; ++q) best[q] = std::max(best[q], len);
            i = j;
        }
    }
    return best;
}

// window-clipped and full anchor runs, blended by ANCHOR_FULL_RUNS.
double window_anchor_length(const std::vector<Row>& rows, const Vec& times, double center,
                            const std::vector<int>& full_runs) {
    const size_t lo = first_not_before(times, center - WINDOW_MS);
    const size_t hi = first_after(times, center + WINDOW_MS);
    if (hi - lo < 2) return 1.0;
    int best = 1;
    std::array<int, 4> streak = {0, 0, 0, 0};
    for (size_t i = lo; i < hi; ++i)
        for (int c = 0; c < 4; ++c) {
            streak[c] = rows[i].mask & (1u << c) ? streak[c] + 1 : 0;
            best = std::max(best, streak[c]);
        }
    int full = 1;
    for (size_t i = lo; i < hi; ++i) full = std::max(full, full_runs[i]);
    return (1.0 - ANCHOR_FULL_RUNS) * static_cast<double>(best) +
           ANCHOR_FULL_RUNS * static_cast<double>(full);
}

// a run of jumps entered and left by jacks is jack content.
std::vector<char> forced_jack_transitions(const std::vector<Row>& rows) {
    size_t n = rows.size();
    std::vector<char> forced(n == 0 ? 0 : n - 1, 0);
    size_t i = 0;
    while (i < n) {
        if (rows[i].size != 2) { ++i; continue; }
        size_t start = i, end = i;
        while (end + 1 < n && rows[end + 1].size == 2) ++end;
        size_t left = start;
        bool entered = false;
        if (start >= 2 && rows[start - 1].size == 1 && transition_is_jack(rows[start - 2], rows[start - 1])) {
            left = start - 1; entered = true;
        } else if (start >= 1 && transition_is_jack(rows[start - 1], rows[start])) {
            left = start; entered = true;
        }
        size_t right = end;
        bool exited = false;
        if (end + 2 < n && rows[end + 1].size == 1 && transition_is_jack(rows[end + 1], rows[end + 2])) {
            right = end + 1; exited = true;
        } else if (end + 1 < n && transition_is_jack(rows[end], rows[end + 1])) {
            right = end; exited = true;
        }
        if (entered && exited)
            for (size_t j = left; j < right && j < forced.size(); ++j) forced[j] = 1;
        i = end + 1;
    }
    return forced;
}

// up to STREAM_BRACKET_MAX stream transitions between two jacks belong to the jack.
std::vector<char> forced_bracketed_stream_transitions(const std::vector<Row>& rows,
                                                      const std::vector<char>& is_jack) {
    size_t n = rows.size();
    std::vector<char> forced(n == 0 ? 0 : n - 1, 0);
    if (n < 2) return forced;
    size_t i = 0, m = n - 1;
    while (i < m) {
        if (is_jack[i]) { ++i; continue; }
        size_t j = i;
        while (j < m && !is_jack[j]) ++j;
        bool left_jack = (i >= 1) && is_jack[i - 1];
        bool right_jack = (j < m) && is_jack[j];
        if (left_jack && right_jack && static_cast<int>(j - i) <= STREAM_BRACKET_MAX)
            for (size_t q = i; q < j; ++q) forced[q] = 1;
        i = j;
    }
    return forced;
}

// rows with grace-linked neighbours merged, and each merged transition's grace weight.
struct GraceRows {
    std::vector<Row> rows;
    Vec weight;
    bool any = false;
};

GraceRows grace_merge(const std::vector<Row>& rows) {
    GraceRows g;
    const size_t n = rows.size();
    if (n < 2) { g.rows = rows; return g; }
    const size_t m = n - 1;
    Vec gap(m), deg(m, 0.0);
    for (size_t i = 0; i < m; ++i) gap[i] = rows[i + 1].t - rows[i].t;
    for (size_t i = 0; i < m; ++i) {
        double before = 0.0, after = 0.0;
        for (size_t k = 1; k <= static_cast<size_t>(GRACE_REACH); ++k) {
            if (i >= k) before = std::max(before, gap[i - k]);
            if (i + k < m) after = std::max(after, gap[i + k]);
        }
        const double ref = (before > 0.0 && after > 0.0) ? std::min(before, after) : std::max(before, after);
        if (ref <= 0.0) continue;
        const double rel = std::min(std::max(1.0 - (gap[i] / ref) / GRACE_RATIO, 0.0), 1.0);
        const double cap = std::min(std::max((GRACE_MAX_MS - gap[i]) / (GRACE_MAX_MS - GRACE_FULL_MS), 0.0), 1.0);
        deg[i] = rel * cap;
    }
    Vec gdeg;
    for (size_t i = 0; i < n;) {
        Row cur = rows[i];
        double d = 1.0;
        size_t j = i;
        while (j + 1 < n && deg[j] > 0.0 && !(cur.mask & rows[j + 1].mask) &&
               rows[j + 1].t - rows[i].t <= GRACE_GROUP_MAX_MS) {
            cur.mask |= rows[j + 1].mask;
            d = std::min(d, deg[j]);
            ++j;
        }
        cur.size = column_count(cur.mask);
        g.rows.push_back(cur);
        gdeg.push_back(j > i ? d : 0.0);
        if (j > i) g.any = true;
        i = j + 1;
    }
    g.weight.assign(g.rows.size() - 1, 0.0);
    for (size_t k = 0; k + 1 < g.rows.size(); ++k) g.weight[k] = std::max(gdeg[k], gdeg[k + 1]);
    return g;
}

// mean grace degree of the graced transitions in each window.
Vec grace_curve(const GraceRows& g, const Vec& grid) {
    Vec on(g.weight.size());
    for (size_t i = 0; i < on.size(); ++i) on[i] = g.weight[i] > 0.0 ? 1.0 : 0.0;
    const Vec times = row_times(g.rows);
    Vec num = window_avg(grid, steps_to_grid(times, g.weight, grid), WINDOW_MS);
    const Vec den = window_avg(grid, steps_to_grid(times, on, grid), WINDOW_MS);
    for (size_t i = 0; i < num.size(); ++i) num[i] = den[i] > 1e-9 ? num[i] / den[i] : 0.0;
    return num;
}

// distance of the gap ratio from a whole number, 0..1. ratios past TECH_BREAK_RATIO are breaks.
double off_integer(double g_a, double g_b) {
    double lo = std::min(g_a, g_b), hi = std::max(g_a, g_b);
    if (lo <= 0.0) return 0.0;
    double prop = hi / lo;
    if (prop > TECH_BREAK_RATIO) return 0.0;
    double d = std::fabs(prop - std::round(prop));
    d -= TECH_RES_GAIN * prop * std::sqrt(1.0 / (lo * lo) + 1.0 / (hi * hi));
    if (d <= 0.0) return 0.0;
    double s = std::sin(PI * d);
    return s * s;
}

// rectify against a deadband onto 0..1.
double rect(double v, double deadband) {
    if (v <= deadband) return 0.0;
    return (v - deadband) / (1.0 - deadband);
}

// fades out as the pattern slows.
double speed_gate(double gap) {
    if (gap <= TECH_SPEED_FLOOR_MS) return 1.0;
    if (gap >= 2.0 * TECH_SPEED_FLOOR_MS) return 0.0;
    return 1.0 - (gap - TECH_SPEED_FLOOR_MS) / TECH_SPEED_FLOOR_MS;
}

// pairwise rhythm irregularity.
Vec chaos_channel(const Vec& gaps) {
    Vec out(gaps.size(), 0.0);
    for (size_t i = 1; i < gaps.size(); ++i)
        out[i] = rect(off_integer(gaps[i], gaps[i - 1]), TECH_CHAOS_DEADBAND) * speed_gate(gaps[i]);
    return out;
}

// short chosen jack runs inside stream.
Vec minijack_channel(const std::vector<Row>& rows, const Vec& gaps, const std::vector<char>& is_jack) {
    size_t n = gaps.size();
    Vec out(n, 0.0);
    size_t i = 0;
    while (i < n) {
        if (!is_jack[i]) { ++i; continue; }
        size_t j = i;
        while (j < n && is_jack[j]) ++j;
        bool chosen = true;
        for (size_t k = i; k < j; ++k)
            if (!minijack_pair(rows[k], rows[k + 1])) { chosen = false; break; }
        bool flanked = (i >= 1) && !is_jack[i - 1] && (j < n) && !is_jack[j];
        if (chosen && flanked && static_cast<int>(j - i) <= TECH_MINIJACK_MAX_RUN) {
            double change = std::max(off_integer(gaps[i], gaps[i - 1]), off_integer(gaps[j - 1], gaps[j]));
            double v = TECH_MINIJACK_FLOOR + (1.0 - TECH_MINIJACK_FLOOR) * change;
            const size_t lo = i >= TECH_BALANCE_ROWS ? i - TECH_BALANCE_ROWS : 0;
            const size_t hi = std::min(n, j + TECH_BALANCE_ROWS);
            double js = 0.0;
            for (size_t q = lo; q < hi; ++q) js += is_jack[q] ? 1.0 : 0.0;
            const double jf = js / static_cast<double>(hi - lo);
            const double keep = std::min(std::max((TECH_MINIJACK_JACK_HI - jf) /
                                                  (TECH_MINIJACK_JACK_HI - TECH_MINIJACK_JACK_LO), 0.0), 1.0);
            for (size_t k = i; k < j; ++k) out[k] = v * keep * speed_gate(gaps[k]);
        }
        i = j;
    }
    return out;
}

// hand and column load skew, gated by how little is jack.
Vec balance_channel(const std::vector<Row>& rows, const std::vector<char>& is_jack) {
    size_t n = rows.empty() ? 0 : rows.size() - 1;
    Vec out(n, 0.0);
    if (n < 1) return out;
    const double log4 = std::log(4.0);
    constexpr size_t R = TECH_BALANCE_ROWS;
    for (size_t i = 0; i < n; ++i) {
        const size_t lo = i + 1 >= R ? i + 1 - R : 0;
        const size_t hi = std::min(i + R + 1, rows.size());
        std::array<double, 4> cnt{};
        double total = 0.0;
        for (size_t r = lo; r < hi; ++r)
            for (int c = 0; c < 4; ++c)
                if (rows[r].mask & (1u << c)) { cnt[c] += 1.0; total += 1.0; }
        const double gate = std::min(static_cast<double>(hi - lo) / static_cast<double>(2 * R), 1.0);
        if (gate <= 0.0 || total <= 0.0) continue;
        double hand = std::fabs((cnt[0] + cnt[1]) - (cnt[2] + cnt[3])) / total;
        double entropy = 0.0;
        for (int c = 0; c < 4; ++c)
            if (cnt[c] > 0.0) { double p = cnt[c] / total; entropy += -p * std::log(p); }
        double finger = 1.0 - entropy / log4;
        const size_t t_hi = std::min(hi, n);
        double js = 0.0;
        for (size_t q = lo; q < t_hi; ++q) js += is_jack[q] ? 1.0 : 0.0;
        const double jack_frac = t_hi > lo ? js / static_cast<double>(t_hi - lo) : 0.0;
        out[i] = rect(0.5 * (hand + finger), TECH_BALANCE_DEADBAND) * (1.0 - jack_frac) * gate;
    }
    return out;
}

Vec technical_raw(const std::vector<Row>& rows, const std::vector<char>& is_jack) {
    size_t n = rows.empty() ? 0 : rows.size() - 1;
    Vec out(n, 0.0);
    if (n < 1) return out;
    Vec gaps(n);
    for (size_t i = 0; i < n; ++i) gaps[i] = rows[i + 1].t - rows[i].t;
    Vec ch = chaos_channel(gaps);
    Vec mj = minijack_channel(rows, gaps, is_jack);
    Vec ba = balance_channel(rows, is_jack);
    for (size_t i = 0; i < n; ++i)
        out[i] = TECH_CHAOS_W * ch[i] + TECH_MINIJACK_W * mj[i] + TECH_BALANCE_W * ba[i];
    return out;
}

// each transition's memberships, held on the grid and averaged over +-WINDOW_MS.
Memberships memberships_of(const std::vector<Row>& rows, const Vec& grid) {
    Memberships memb;
    for (Vec& m : memb) m.assign(grid.size(), 0.0);
    if (rows.size() < 2) return memb;
    const size_t n = rows.size() - 1;
    const Vec times = row_times(rows);
    const std::vector<char> jack = jack_flags(rows);
    const std::vector<char> forced = forced_jack_transitions(rows);
    const std::vector<char> bracketed = forced_bracketed_stream_transitions(rows, jack);
    const std::vector<char> hs = handstream_flags(rows);
    const std::vector<int> full_runs = anchor_run_lengths(rows);
    Memberships per;
    for (Vec& m : per) m.assign(n, 0.0);
    per[SS_TECHNICAL] = technical_raw(rows, jack);
    for (size_t i = 0; i < n; ++i) {
        const Row& a = rows[i];
        const Row& b = rows[i + 1];
        const double center = 0.5 * (a.t + b.t);
        if (jack[i] || forced[i] || bracketed[i] || std::max(a.size, b.size) >= 4) {
            const double anchor = window_anchor_length(rows, times, center, full_runs);
            const double density = window_density(rows, times, center);
            const double blend = sharpen_split(
                JACK_ANCHOR_W * sigmoid(JACK_ANCHOR_SLOPE * (anchor - JACK_ANCHOR_MID)) +
                (1.0 - JACK_ANCHOR_W) * sigmoid(JACK_DENSITY_SLOPE * (density - JACK_DENSITY_MID)));
            per[SS_CHORDJACK][i] = blend;
            per[SS_SPEEDJACK][i] = 1.0 - blend;
        } else if (hs[i]) {
            per[SS_HANDSTREAM][i] = 1.0;
        } else {
            const double jd = window_jump_density(rows, times, center);
            const double js = sigmoid(JS_SPLIT_SLOPE * (jd - STREAM_JS_DENSITY));
            per[SS_JUMPSTREAM][i] = js;
            per[SS_STREAM][i] = 1.0 - js;
        }
    }
    for (int k = 0; k < NUM_PATTERN_SKILLSETS; ++k)
        memb[k] = window_avg(grid, steps_to_grid(times, per[k], grid), WINDOW_MS);
    // saturate technical so a window dense with awkwardness reads near 1.
    for (double& v : memb[SS_TECHNICAL]) v = 1.0 - std::exp(-TECH_GAIN * v);
    return memb;
}

// the six classifier memberships. only head times are read. graced patterns split
// between their raw and merged readings.
Memberships pattern_memberships(const std::vector<Note>& notes, const Vec& grid) {
    const std::vector<Row> rows = build_rows(notes);
    Memberships gm = memberships_of(rows, grid);
    const GraceRows grace = grace_merge(rows);
    if (grace.any) {
        const Vec gw = grace_curve(grace, grid);
        const Memberships gg = memberships_of(grace.rows, grid);
        for (int k = 0; k < NUM_PATTERN_SKILLSETS; ++k)
            for (size_t i = 0; i < gm[k].size(); ++i) gm[k][i] += gw[i] * (gg[k][i] - gm[k][i]);
    }
    return gm;
}

// release membership from the hold costs, normalised to the map's own maximum.
Vec release_membership(const std::vector<Note>& notes, const Vec& rel, const Vec& cross_held,
                       const Vec& weight_held, const Vec& grid) {
    // release fills whole tail-to-tail gaps, so it is gated to where a real hold is.
    Vec mask(grid.size(), 0.0);
    for (const Note& n : notes) {
        if (n.t < 0 || n.t - n.h <= LN_TAP_MS) continue;
        const size_t lo = first_not_before(grid, n.h);
        const size_t hi = first_after(grid, n.t + RELEASE_TAIL_MS);
        for (size_t i = lo; i < hi; ++i) mask[i] = 1.0;
    }
    mask = window_avg(grid, mask, WINDOW_MS);
    Vec gated(grid.size(), 0.0);
    double gmax = 0.0;
    for (size_t i = 0; i < gated.size(); ++i) {
        gated[i] = rel[i] * mask[i] + cross_held[i] + RELEASE_MEMB_WEIGHT * weight_held[i];
        gmax = std::max(gmax, gated[i]);
    }
    if (gmax > 0.0) for (double& v : gated) v /= gmax;
    return gated;
}

// the full algorithm for one map: technique and hold cost per hand, stamina per hand,
// hands combined, then aggregation and skillsets. rated once with long notes and once
// with every long note read as a tap (rice).

// ratings are the displayed numbers.
struct Analysis {
    double rating = 0.0;
    double rating_rice = 0.0;   // every long note read as a tap
    std::array<double, NUM_SKILLSETS> skillsets{};
    Vec grid;                   // ms
    Vec graph;                  // strain without stamina, in points
};

constexpr double GRID_MS = 25.0;

// same column within this many ms is one note mapped twice.
constexpr int DUP_MS = 10;

// hand strain = technique + LN_W * hold cost^LN_X
constexpr double LN_W = 3.997;
constexpr double LN_X = 1.3275;

// notes: sorted by (h, k), non-negative ms with the rate applied. a note repeating its
// column within DUP_MS is merged into the one before.
Analysis analyse(const std::vector<Note>& notes) {
    Analysis a;
    std::vector<Note> seq;
    std::array<int, 4> last = {-1, -1, -1, -1};
    for (const Note& n : notes) {
        const int j = last[n.k];
        if (j >= 0 && n.h - seq[j].h <= DUP_MS) { seq[j].t = std::max(seq[j].t, n.t); continue; }
        last[n.k] = static_cast<int>(seq.size());
        seq.push_back(n);
    }
    if (seq.size() < 2) return a;

    std::array<std::vector<Note>, 4> by_col;
    std::array<std::vector<Note>, 2> by_hand;
    int end = 0;
    bool any_ln = false;
    for (const Note& n : seq) {
        by_col[n.k].push_back(n);
        by_hand[n.k / 2].push_back(n);
        end = std::max({end, n.h, n.t});
        any_ln = any_ln || n.t >= 0;
    }
    Vec& grid = a.grid;
    for (double t = 0.0; t <= end + 1000.0; t += GRID_MS) grid.push_back(t);
    const double x = hit_window_unit();

    // pressing strain, and the rice strain (held fingers stay busy, so it differs).
    std::array<Vec, 2> able, strain_rice;
    std::array<Vec, 2> strain = technique_strain(x, by_col, grid, able, strain_rice);

    // hold cost per hand, added to its pressing strain.
    std::array<HoldCosts, 2> holds;
    for (int h = 0; h < 2; ++h) {
        std::array<std::vector<Note>, 4> cols;
        for (int c = 2 * h; c < 2 * h + 2; ++c) cols[c] = by_col[c];
        holds[h] = hold_costs(x, cols, by_hand[h], grid);
        for (size_t i = 0; i < grid.size(); ++i) {
            const double l = holds[h].rel[i] + holds[h].weight[i] + holds[h].cross[i];
            if (l > 0.0) strain[h][i] += LN_W * std::pow(l, LN_X);
        }
    }
    const std::array<Vec, 2> rested = strain;

    // stamina per hand.
    const std::array<Vec, 2> w = {note_weights(grid, by_hand[0]), note_weights(grid, by_hand[1])};
    apply_stamina(grid, strain, w);
    if (any_ln) apply_stamina(grid, strain_rice, w);

    // combine and aggregate.
    const std::vector<Row> rows = build_rows(seq);
    const Vec sync = hand_sync(able, rows, grid);
    const Vec all = combine_hands(strain, sync);
    const Vec all_rice = any_ln ? combine_hands(strain_rice, sync) : all;
    a.graph = combine_hands(rested, sync);
    for (double& v : a.graph) v = strain_points(v);
    const Vec weights = note_weights(grid, seq);
    const double tn = static_cast<double>(seq.size());
    const BinnedStrain pd = bin_strain(all);
    const BinnedStrain pd_rice = any_ln ? bin_strain(all_rice) : pd;
    const double raw = rating_from_bins(pd, weights, tn);
    const double raw_rice = any_ln ? rating_from_bins(pd_rice, weights, tn) : raw;
    a.rating = rating_points(raw);
    a.rating_rice = rating_points(raw_rice);

    // skillsets: the classifier six read the rice curve, release the full one.
    std::array<Vec, NUM_SKILLSETS> memberships;
    Memberships pm = pattern_memberships(seq, grid);
    for (int k = 0; k < NUM_PATTERN_SKILLSETS; ++k) memberships[k] = std::move(pm[k]);
    Vec rel(grid.size(), 0.0), cross_held(grid.size(), 0.0), weight_held(grid.size(), 0.0);
    for (const HoldCosts& hc : holds)
        for (size_t i = 0; i < grid.size(); ++i) {
            rel[i] += hc.rel[i];
            cross_held[i] += hc.cross_held[i];
            weight_held[i] += hc.weight_held[i];
        }
    memberships[SS_RELEASE] = release_membership(seq, rel, cross_held, weight_held, grid);

    const double total_w = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int s = 0; s < NUM_SKILLSETS; ++s) {
        Vec ws(grid.size(), 0.0);
        for (size_t i = 0; i < grid.size(); ++i) ws[i] = weights[i] * std::max(memberships[s][i], 0.0);
        a.skillsets[s] = rating_points(
            skillset_rating(s == SS_RELEASE ? pd : pd_rice, ws, total_w, tn, raw));
    }
    return a;
}

}  // namespace xdc1

namespace chart {

struct RawNote {
    int k;
    long long h, t;   // t is -1 for a tap
};

// 4k .osu hit objects. column is floor(x) / 128; type bit 7 marks a hold.
std::vector<RawNote> parse_osu(std::istream& in) {
    std::vector<RawNote> notes;
    std::string line;
    int keys = -1;
    bool in_objects = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!in_objects) {
            const size_t b = line.find_first_not_of(" \t");
            if (b != std::string::npos && line.compare(b, 11, "CircleSize:") == 0) {
                try {
                    const int n = static_cast<int>(std::lround(std::stod(line.substr(b + 11))));
                    if (n >= 1) keys = n;
                } catch (const std::exception&) {
                }
            }
            if (line.find("[HitObjects]") != std::string::npos) {
                if (keys != 4)
                    throw std::runtime_error("only 4K charts are supported (CircleSize " +
                                             std::to_string(keys) + ")");
                in_objects = true;
            }
            continue;
        }
        if (line.empty()) continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        for (std::string tok; std::getline(ss, tok, ',');) f.push_back(tok);
        if (f.size() < 6) continue;
        const int k = static_cast<int>(std::stod(f[0])) / (512 / keys);
        if (k < 0 || k >= keys)
            throw std::runtime_error("hit object at x=" + f[0] + " is outside the 4 columns");
        const long long h = std::stoll(f[2]);
        const int type = std::stoi(f[3]);
        const long long t = (type & 128) ? std::stoll(f[5].substr(0, f[5].find(':'))) : -1;
        notes.push_back({k, h, t});
    }
    if (!in_objects) throw std::runtime_error("no [HitObjects] section");
    return notes;
}

// one "column head tail" per line. blank lines and # comments are skipped.
std::vector<RawNote> parse_plain(std::istream& in) {
    std::vector<RawNote> notes;
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        const size_t b = line.find_first_not_of(" \t\r");
        if (b == std::string::npos || line[b] == '#') continue;
        std::istringstream ls(line);
        RawNote n{};
        if (!(ls >> n.k >> n.h >> n.t))
            throw std::runtime_error("line " + std::to_string(lineno) +
                                     ": expected \"column head tail\"");
        if (n.k < 0 || n.k > 3)
            throw std::runtime_error("line " + std::to_string(lineno) + ": column must be 0-3");
        notes.push_back(n);
    }
    return notes;
}

// times become floor(t / rate). a tail before its head is read as a tap.
std::vector<xdc1::Note> notes_at(const std::vector<RawNote>& raw, double rate) {
    std::vector<xdc1::Note> seq;
    seq.reserve(raw.size());
    for (const RawNote& r : raw) {
        long long h = r.h, t = r.t;
        if (rate != 1.0) {
            h = static_cast<long long>(std::floor(h / rate));
            if (t >= 0) t = static_cast<long long>(std::floor(t / rate));
        }
        if (h < 0) throw std::runtime_error("negative note time");
        if (t >= 0 && t < h) t = -1;
        seq.push_back({r.k, static_cast<int>(h), static_cast<int>(t)});
    }
    std::sort(seq.begin(), seq.end(), [](const xdc1::Note& a, const xdc1::Note& b) {
        return a.h != b.h ? a.h < b.h : a.k < b.k;
    });
    return seq;
}

}  // namespace chart

#ifndef XDC1_NO_MAIN

namespace {

void print_array(const char* name, const xdc1::Vec& v) {
    std::printf(",\n  \"%s\": [", name);
    for (size_t i = 0; i < v.size(); ++i) std::printf(i ? ",%.6g" : "%.6g", v[i]);
    std::printf("]");
}

int usage() {
    std::fprintf(stderr, "usage: xdc1 [--rate R] [--graph] [chart.osu | -]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    double rate = 1.0;
    bool graph = false;
    const char* path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rate") && i + 1 < argc) {
            char* end = nullptr;
            rate = std::strtod(argv[++i], &end);
            if (*end || !(rate > 0.0)) return usage();
        } else if (!std::strcmp(argv[i], "--graph")) {
            graph = true;
        } else if (!path && (argv[i][0] != '-' || !std::strcmp(argv[i], "-"))) {
            path = argv[i];
        } else {
            return usage();
        }
    }

    try {
        std::string text;
        if (!path || !std::strcmp(path, "-")) {
            text.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
        } else {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error(std::string("cannot open ") + path);
            text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        std::istringstream in(text);
        const std::vector<chart::RawNote> raw = text.find("[HitObjects]") != std::string::npos
            ? chart::parse_osu(in) : chart::parse_plain(in);
        if (raw.empty()) throw std::runtime_error("the chart has no notes");

        const xdc1::Analysis a = xdc1::analyse(chart::notes_at(raw, rate));

        std::printf("{\n  \"rating\": %.4f,\n  \"rating_rice\": %.4f,\n  \"skillsets\": {",
                    a.rating, a.rating_rice);
        for (int s = 0; s < xdc1::NUM_SKILLSETS; ++s)
            std::printf("%s\"%s\": %.4f", s ? ", " : "", xdc1::SKILLSETS[s], a.skillsets[s]);
        std::printf("}");
        if (graph) {
            print_array("grid", a.grid);
            print_array("graph", a.graph);
        }
        std::printf("\n}\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xdc1: %s\n", e.what());
        return 1;
    }
    return 0;
}

#endif  // XDC1_NO_MAIN
