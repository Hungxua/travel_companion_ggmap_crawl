"""
ITINERARY GENERATOR V2 — Tour Guide Approach
4 bước như một tour guide thực sự lên kế hoạch:

  Step 1: Cluster — Nhìn bản đồ, tự động khoanh vùng theo khoảng cách
  Step 2: Route   — Nối các cụm thành tuyến 1 chiều, không zigzag
  Step 3: Assign  — Chia cụm vào từng ngày theo ngân sách thời gian
  Step 4: Detail  — Lên chi tiết từng ngày trong phạm vi cụm đã xác định
"""

import json
import math
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum


# ============================================================================
# DATA MODELS (unchanged)
# ============================================================================

@dataclass
class Coordinates:
    lat: float
    lng: float


@dataclass
class Place:
    id: str
    name: str
    poi_type: str
    coordinates: Coordinates
    district: str
    visit_duration_minutes: int
    vibe_scores: Dict[str, float]
    companion_scores: Dict[str, float]
    priority_score: float
    google_maps_rating: float
    google_maps_reviews_count: int
    tips: List[str]
    opening_hours: Optional[str] = None
    night_danger: Optional[bool] = False
    must_visit: Optional[bool] = False
    warnings: Optional[List[Dict]] = None
    meal_type: Optional[str] = None
    avg_spending: Optional[int] = None
    check_in_time: Optional[str] = None
    check_out_time: Optional[str] = None


@dataclass
class Cluster:
    """Một cụm địa điểm gần nhau về địa lý"""
    id: int
    centroid: Coordinates
    attractions: List[Place]
    total_visit_minutes: int  # Tổng thời gian thăm quan (chưa tính di chuyển nội bộ)
    label: str = ""           # Tên tự gán (ví dụ: "Khu vực Đồng Văn")


@dataclass
class DayPlan:
    """Kế hoạch 1 ngày sau khi assign clusters"""
    day_number: int
    clusters: List[Cluster]
    start_location: Coordinates
    end_location: Coordinates


@dataclass
class ScheduleItem:
    time: str
    type: str   # 'attraction' | 'food' | 'accommodation' | 'transport'
    place: Optional[Dict] = None
    duration_minutes: Optional[int] = None
    travel_time_minutes: Optional[int] = None
    distance_km: Optional[float] = None
    meal_type: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class DaySchedule:
    day: int
    date: str
    title: str
    schedule: List[ScheduleItem]
    summary: Dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(coord1: Coordinates, coord2: Coordinates) -> float:
    """Khoảng cách thẳng (km) theo công thức Haversine"""
    R = 6371
    lat1, lat2 = math.radians(coord1.lat), math.radians(coord2.lat)
    dlat = math.radians(coord2.lat - coord1.lat)
    dlng = math.radians(coord2.lng - coord1.lng)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_time_minutes(c1: Coordinates, c2: Coordinates, mountain: bool = True) -> int:
    """Ước tính thời gian di chuyển (phút)"""
    d = haversine_distance(c1, c2)
    actual = d * (1.4 if mountain else 1.1)
    speed = 25 if mountain else 40
    return max(1, int(actual / speed * 60))


def parse_time(s: str) -> datetime:
    return datetime.strptime(s, "%H:%M")


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def add_min(dt: datetime, m: int) -> datetime:
    return dt + timedelta(minutes=m)


def centroid(places: List[Place]) -> Coordinates:
    """Tọa độ trung tâm của một tập điểm"""
    lats = [p.coordinates.lat for p in places]
    lngs = [p.coordinates.lng for p in places]
    return Coordinates(sum(lats) / len(lats), sum(lngs) / len(lngs))


# ============================================================================
# STEP 1 — CLUSTER: Tự động nhóm điểm gần nhau (DBSCAN-style, không cần sklearn)
# ============================================================================

def cluster_attractions(attractions: List[Place],
                        eps_km: float = 18.0,
                        min_samples: int = 1) -> List[Cluster]:
    """
    Thuật toán gom cụm đơn giản kiểu DBSCAN không dùng thư viện ngoài.
    
    eps_km   : hai điểm cách nhau <= eps_km thì coi là "láng giềng"
    min_samples: tối thiểu 1 điểm để tạo cụm (mọi điểm đều thuộc 1 cụm nào đó)

    Trả về: list[Cluster], mỗi cluster là 1 cụm địa lý
    """
    n = len(attractions)
    labels = [-1] * n   # -1 = chưa gán
    cluster_id = 0

    def neighbors(idx: int) -> List[int]:
        return [
            j for j in range(n)
            if j != idx and haversine_distance(
                attractions[idx].coordinates,
                attractions[j].coordinates
            ) <= eps_km
        ]

    for i in range(n):
        if labels[i] != -1:
            continue
        nbrs = neighbors(i)
        if len(nbrs) < min_samples - 1:
            # Điểm isolated — tự thành cụm riêng
            labels[i] = cluster_id
            cluster_id += 1
            continue
        labels[i] = cluster_id
        queue = list(nbrs)
        while queue:
            j = queue.pop(0)
            if labels[j] == -1:
                labels[j] = cluster_id
                j_nbrs = neighbors(j)
                if len(j_nbrs) >= min_samples - 1:
                    queue.extend([x for x in j_nbrs if labels[x] == -1])
            elif labels[j] != cluster_id:
                labels[j] = cluster_id  # merge
        cluster_id += 1

    # Group by label
    groups: Dict[int, List[Place]] = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(lbl, []).append(attractions[i])

    clusters = []
    for cid, places in sorted(groups.items()):
        c = centroid(places)
        total_min = sum(p.visit_duration_minutes for p in places)
        # Label dựa theo district xuất hiện nhiều nhất
        district_count: Dict[str, int] = {}
        for p in places:
            district_count[p.district] = district_count.get(p.district, 0) + 1
        main_district = max(district_count, key=district_count.get)
        clusters.append(Cluster(
            id=cid,
            centroid=c,
            attractions=places,
            total_visit_minutes=total_min,
            label=f"Khu vực {main_district}"
        ))

    return clusters


# ============================================================================
# STEP 2 — ROUTE: Nối các cụm thành tuyến 1 chiều (Nearest Neighbor TSP)
# ============================================================================

def order_clusters(clusters: List[Cluster],
                   start: Coordinates) -> List[Cluster]:
    """
    Sắp xếp thứ tự các cụm thành tuyến đường tối ưu (không zigzag).
    Nearest Neighbor: từ vị trí hiện tại luôn chọn cụm gần nhất chưa thăm.
    """
    print(f"   📍 Điểm xuất phát: ({start.lat:.4f}, {start.lng:.4f})")
    print(f"   Khoảng cách từ xuất phát đến từng cụm:")
    for c in clusters:
        d = haversine_distance(start, c.centroid)
        print(f"      {c.label}: {d:.1f}km")

    unvisited = list(clusters)
    ordered = []
    current = start
    step = 0

    while unvisited:
        distances = {c.id: haversine_distance(current, c.centroid) for c in unvisited}
        nearest = min(unvisited, key=lambda c: distances[c.id])
        from_label = "xuất phát" if step == 0 else ordered[-1].label

        print(f"\n   Bước {step + 1} — từ [{from_label}], xét các cụm còn lại:")
        for c in sorted(unvisited, key=lambda c: distances[c.id]):
            marker = "✅ chọn" if c.id == nearest.id else "      "
            print(f"      {marker} {c.label}: {distances[c.id]:.1f}km")

        ordered.append(nearest)
        unvisited.remove(nearest)
        current = nearest.centroid
        step += 1

    return ordered


# ============================================================================
# STEP 3 — ASSIGN: Chia cụm vào từng ngày theo ngân sách thời gian
# ============================================================================

# Thời gian mỗi ngày dành cho tham quan (phút)
FULL_DAY_BUDGET = 480    # 8 giờ thực tế (di chuyển + tham quan)


def assign_clusters_to_days(ordered_clusters: List[Cluster],
                             total_days: int,
                             start: Coordinates) -> List[DayPlan]:
    """
    Chia danh sách cụm (đã sắp thứ tự) vào từng ngày.

    Nguyên tắc:
    - Mọi ngày đều có ngân sách bằng nhau (FULL_DAY_BUDGET)
    - Không bẻ ngược thứ tự cụm
    - overhead tính động = travel_time thực từ vị trí trước đến centroid cụm
    """
    budgets = [FULL_DAY_BUDGET] * total_days

    # Greedy fill: lần lượt nhét cụm vào ngày đang xét
    day_clusters: List[List[Cluster]] = [[] for _ in range(total_days)]
    day_used: List[int] = [0] * total_days
    current_day = 0
    # prev_centroid theo dõi vị trí cuối của ngày hiện tại để tính travel đến cụm tiếp
    prev_centroid = start

    for cluster in ordered_clusters:
        # Overhead = travel thực từ vị trí trước đến centroid cụm này
        overhead = travel_time_minutes(prev_centroid, cluster.centroid)
        # Thời gian tham quan: avg × tối đa 4 điểm
        avg_duration = cluster.total_visit_minutes / max(len(cluster.attractions), 1)
        slots = min(len(cluster.attractions), 4)
        needed = int(avg_duration * slots) + overhead

        fits = day_used[current_day] + needed <= budgets[current_day]
        print(f"\n   {cluster.label} — ước tính {needed}ph "
              f"(avg {avg_duration:.0f}ph × {slots} điểm + {overhead}ph travel overhead)")

        if fits:
            print(f"      → fit vào Ngày {current_day + 1} "
                  f"(đã dùng {day_used[current_day]}ph + {needed}ph = "
                  f"{day_used[current_day] + needed}ph / {budgets[current_day]}ph budget)")
            day_clusters[current_day].append(cluster)
            day_used[current_day] += needed
        else:
            print(f"      → Ngày {current_day + 1} không đủ chỗ "
                  f"({day_used[current_day]}ph + {needed}ph > {budgets[current_day]}ph), "
                  f"chuyển sang Ngày {current_day + 2}")
            current_day += 1
            if current_day >= total_days:
                current_day = total_days - 1
            day_clusters[current_day].append(cluster)
            day_used[current_day] += needed

        # Cập nhật prev_centroid = centroid cụm vừa assign (để tính overhead cụm tiếp)
        prev_centroid = cluster.centroid

    # Build DayPlan, tính start/end location
    plans: List[DayPlan] = []
    prev_end = start

    for d_idx, clusters in enumerate(day_clusters):
        day_num = d_idx + 1
        day_start = prev_end

        if clusters:
            # Điểm cuối ngày = centroid của cụm cuối cùng
            day_end = clusters[-1].centroid
        else:
            day_end = day_start

        plans.append(DayPlan(
            day_number=day_num,
            clusters=clusters,
            start_location=day_start,
            end_location=day_end
        ))
        prev_end = day_end

    return plans


# ============================================================================
# STEP 4 — DETAIL: Lên chi tiết từng ngày trong phạm vi cụm
# ============================================================================

class DetailScheduler:
    """
    Chỉ làm việc với những điểm thuộc cụm của ngày đó.
    Phạm vi đã thu hẹp từ 50 → 8-10 điểm → scoring chính xác hơn.
    """

    MAX_ATTRACTIONS_PER_DAY = 4
    DAILY_START_TIME = "08:00"

    def __init__(self, food_places: List[Place],
                 accommodations: List[Place],
                 user_prefs: Dict):
        self.food_places = food_places
        self.accommodations = accommodations
        self.prefs = user_prefs
        self.visited_ids: set = set()

    # ------------------------------------------------------------------
    # Scoring (user-preference aware)
    # ------------------------------------------------------------------

    def score_place(self, place: Place) -> float:
        interests = self.prefs.get('interests', [])
        companion = self.prefs.get('companions', 'solo')

        # Vibe match
        interest_score = sum(
            place.vibe_scores.get(i, 0) * 3.0 for i in interests
        )
        interest_score += sum(
            s * 0.5 for k, s in place.vibe_scores.items() if k not in interests
        )

        companion_bonus = place.companion_scores.get(companion, 0.5)
        popularity = (place.google_maps_rating / 5.0) * math.log10(
            place.google_maps_reviews_count + 1
        )
        must_visit_bonus = 2.0 if place.must_visit else 0.0

        return (
            interest_score  * 0.2 +
            companion_bonus * 0.2 +
            place.priority_score * 0.2 +
            popularity      * 0.2 +
            must_visit_bonus * 0.2
        )

    # ------------------------------------------------------------------
    # Greedy selection — chỉ trong pool của cụm ngày đó
    # ------------------------------------------------------------------

    def select_attractions(self, pool: List[Place],
                            start: Coordinates,
                            time_budget: int) -> List[Dict]:
        selected = []
        current_loc = start
        remaining = time_budget
        # visited_ids chỉ track trong ngày này — không dùng self.visited_ids
        day_visited: set = set()

        while len(selected) < self.MAX_ATTRACTIONS_PER_DAY and remaining > 60:
            candidates = []
            for p in pool:
                if p.id in day_visited:
                    continue
                tt = travel_time_minutes(current_loc, p.coordinates)
                dist = haversine_distance(current_loc, p.coordinates)
                total = tt + p.visit_duration_minutes
                if total > remaining:
                    continue

                q = self.score_place(p)
                dist_penalty = dist / 100
                time_penalty = total / 240
                must_bonus = 2.0 if p.must_visit else 0.0

                score = (q                  * 0.25 +
                         (1 - dist_penalty) * 0.25 +
                         (1 - time_penalty) * 0.25 +
                         must_bonus         * 0.25)

                candidates.append({
                    'place': p,
                    'travel_time': tt,
                    'distance_km': dist,
                    'total_time': total,
                    'score': score
                })

            if not candidates:
                break

            # Log: toàn bộ ứng viên, sort theo score
            candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
            best = candidates_sorted[0]
            rank = len(selected) + 1
            print(f"\n      Chọn điểm #{rank} (còn {remaining}ph):")
            for i, cand in enumerate(candidates_sorted[:5]):  # top 5
                p = cand['place']
                marker = "✅" if i == 0 else f"  {i+1}."
                q = self.score_place(p)
                print(f"         {marker} {p.name}  score={cand['score']:.3f} "
                      f"(quality={q:.2f}, dist={cand['distance_km']:.1f}km, "
                      f"travel={cand['travel_time']}ph, visit={p.visit_duration_minutes}ph)")
            if len(candidates_sorted) > 5:
                print(f"         ... và {len(candidates_sorted) - 5} ứng viên khác")

            selected.append(best)
            day_visited.add(best['place'].id)
            current_loc = best['place'].coordinates
            remaining -= best['total_time']

        return selected

    # ------------------------------------------------------------------
    # Accommodation: tìm chỗ ngủ gần điểm cuối ngày
    # ------------------------------------------------------------------

    def find_accommodation(self, near: Coordinates,
                            is_last_day: bool) -> Optional[Place]:
        if is_last_day:
            return None
        candidates = sorted(
            self.accommodations,
            key=lambda a: (
                -0.6 * (1 - haversine_distance(near, a.coordinates) / 50) +
                -0.4 * (a.google_maps_rating / 5.0)
            )
        )
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # Food: tìm quán ăn theo loại bữa, gần location hiện tại
    # ------------------------------------------------------------------

    def find_food(self, near: Coordinates, meal_type: str) -> Optional[Place]:
        pool = [
            f for f in self.food_places
            if f.meal_type in (meal_type, 'all_day')
        ]
        if not pool:
            pool = [f for f in self.food_places if f.meal_type == 'all_day']
        if not pool:
            return None
        return min(pool, key=lambda f: haversine_distance(near, f.coordinates))

    # ------------------------------------------------------------------
    # Build full schedule for one day
    # ------------------------------------------------------------------

    def build_day_schedule(self, plan: DayPlan,
                            total_days: int,
                            start_date: str) -> DaySchedule:
        items: List[ScheduleItem] = []
        now = parse_time(self.DAILY_START_TIME)
        loc = plan.start_location

        # Date string
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        date_str = (start_dt + timedelta(days=plan.day_number - 1)).strftime("%Y-%m-%d")

        # Flatten all attractions từ clusters của ngày này
        pool: List[Place] = []
        for cluster in plan.clusters:
            pool.extend(cluster.attractions)

        # Time budget: trừ cả 3 bữa — mọi ngày đều có breakfast tại địa phương
        time_budget = FULL_DAY_BUDGET - (45 + 60 + 60)     # breakfast + lunch + dinner

        # ── Breakfast (mọi ngày) ─────────────────────────────────────────
        bf = self.find_food(loc, 'breakfast')
        if bf:
            tt = travel_time_minutes(loc, bf.coordinates)
            now = add_min(now, tt)
            items.append(ScheduleItem(
                time=fmt_time(now), type='food', meal_type='breakfast',
                place=self._pdict(bf), duration_minutes=45,
                travel_time_minutes=tt
            ))
            now = add_min(now, 45)
            loc = bf.coordinates

        # ── Chọn attractions từ pool cụm của ngày ──────────────────────
        # Dùng centroid làm start để tính remaining (tránh double count travel liên cụm)
        # nhưng dùng plan.start_location để tính travel thực đến điểm #1 trong timeline
        budget_start = plan.clusters[0].centroid if plan.clusters else loc
        selected = self.select_attractions(pool, budget_start, time_budget)

        lunch_done = False
        first_attraction = True
        for attr in selected:
            p = attr['place']
            dist = attr['distance_km']

            if first_attraction:
                # Điểm #1: travel từ vị trí thực (sau breakfast) đến điểm
                tt = travel_time_minutes(loc, p.coordinates)
                first_attraction = False
            else:
                tt = attr['travel_time']

            # Di chuyển tới điểm
            now = add_min(now, tt)
            items.append(ScheduleItem(
                time=fmt_time(now), type='attraction',
                place=self._pdict(p),
                duration_minutes=p.visit_duration_minutes,
                travel_time_minutes=tt,
                distance_km=round(haversine_distance(loc, p.coordinates), 1)
            ))
            now = add_min(now, p.visit_duration_minutes)
            loc = p.coordinates

            # Chèn bữa trưa nếu đúng khung giờ 11:00–14:00
            if not lunch_done and 11 <= now.hour <= 14:
                lf = self.find_food(loc, 'lunch')
                if lf:
                    tt_l = travel_time_minutes(loc, lf.coordinates)
                    now = add_min(now, tt_l)
                    items.append(ScheduleItem(
                        time=fmt_time(now), type='food', meal_type='lunch',
                        place=self._pdict(lf), duration_minutes=60,
                        travel_time_minutes=tt_l
                    ))
                    now = add_min(now, 60)
                    loc = lf.coordinates
                    lunch_done = True

        # ── Accommodation ────────────────────────────────────────────────
        acc = self.find_accommodation(loc, plan.day_number == total_days)
        acc_loc = acc.coordinates if acc else loc

        # ── Dinner (tìm gần accommodation) ──────────────────────────────
        df = self.find_food(acc_loc, 'dinner')
        if df:
            tt_d = travel_time_minutes(loc, df.coordinates)
            now = add_min(now, tt_d)
            items.append(ScheduleItem(
                time=fmt_time(now), type='food', meal_type='dinner',
                place=self._pdict(df), duration_minutes=60,
                travel_time_minutes=tt_d
            ))
            now = add_min(now, 60)
            loc = df.coordinates

        # ── Check-in ────────────────────────────────────────────────────
        if acc:
            tt_a = travel_time_minutes(loc, acc.coordinates)
            now = add_min(now, tt_a)
            dist_moved = haversine_distance(plan.start_location, acc.coordinates)
            reason = (
                f"Nghỉ gần điểm cuối (di chuyển {dist_moved:.1f}km từ đầu ngày)"
                if dist_moved > 20
                else f"Nghỉ gần khu vực hiện tại ({dist_moved:.1f}km)"
            )
            items.append(ScheduleItem(
                time=fmt_time(now), type='accommodation',
                place=self._pdict(acc), notes=reason
            ))

        # ── Title ────────────────────────────────────────────────────────
        cluster_labels = [c.label for c in plan.clusters]
        if plan.day_number == 1:
            title = f"TP Hà Giang → {cluster_labels[0] if cluster_labels else 'điểm đến'}"
        elif plan.day_number == total_days:
            title = f"{cluster_labels[0] if cluster_labels else 'điểm đến'} → TP Hà Giang"
        else:
            title = " + ".join(cluster_labels) if cluster_labels else f"Ngày {plan.day_number}"

        # ── Summary ─────────────────────────────────────────────────────
        total_dist = sum(
            i.distance_km for i in items
            if i.distance_km is not None
        )
        elapsed = int((now - parse_time(self.DAILY_START_TIME)).total_seconds() / 60)

        summary = {
            'total_attractions': len(selected),
            'clusters_visited': [c.label for c in plan.clusters],
            'total_distance_km': round(total_dist, 1),
            'total_time_minutes': elapsed,
            'accommodation': acc.name if acc else None
        }

        return DaySchedule(
            day=plan.day_number,
            date=date_str,
            title=title,
            schedule=[asdict(i) for i in items],
            summary=summary
        )

    def _pdict(self, p: Place) -> Dict:
        return {
            'id': p.id,
            'name': p.name,
            'type': p.poi_type,
            'coordinates': {'lat': p.coordinates.lat, 'lng': p.coordinates.lng},
            'rating': p.google_maps_rating,
            'tips': p.tips,
            'warnings': p.warnings or [],
            'avg_spending': p.avg_spending
        }


# ============================================================================
# ORCHESTRATOR — gọi 4 bước theo thứ tự
# ============================================================================

def dict_to_place(data: Dict) -> Place:
    coords = Coordinates(lat=data['coordinates']['lat'],
                         lng=data['coordinates']['lng'])
    return Place(
        id=data['id'],
        name=data['name'],
        poi_type=data['poi_type'],
        coordinates=coords,
        district=data['district'],
        visit_duration_minutes=data.get('visit_duration_minutes', 60),
        vibe_scores=data.get('vibe_scores', {}),
        companion_scores=data.get('companion_scores', {}),
        priority_score=data.get('priority_score', 0.5),
        google_maps_rating=data.get('google_maps_rating', 4.0),
        google_maps_reviews_count=data.get('google_maps_reviews_count', 0),
        tips=data.get('tips', []),
        opening_hours=data.get('opening_hours'),
        night_danger=data.get('night_danger', False),
        must_visit=data.get('must_visit', False),
        warnings=data.get('warnings', []),
        meal_type=data.get('meal_type'),
        avg_spending=data.get('avg_spending'),
        check_in_time=data.get('check_in_time'),
        check_out_time=data.get('check_out_time')
    )


def generate_itinerary(places_data: Dict, user_prefs: Dict) -> Dict:
    """
    Entry point. Thực thi 4 bước và trả về lịch trình hoàn chỉnh.
    """

    # ── Parse data ──────────────────────────────────────────────────────
    dest = places_data['destination']
    attractions = [dict_to_place(p) for p in places_data['places']['attractions']]
    food_places  = [dict_to_place(p) for p in places_data['places']['food']]
    accommodations = [dict_to_place(p) for p in places_data['places']['accommodations']]
    # Điểm bắt đầu cố định: trung tâm TP Hà Giang
    start = Coordinates(lat=22.8233, lng=104.9836)
    total_days = user_prefs.get('days', 3)
    start_date = user_prefs.get('start_date', '2024-03-15')

    # ══════════════════════════════════════════════════════════════════
    # STEP 1 — CLUSTER: Nhìn bản đồ, tự động khoanh vùng
    # ══════════════════════════════════════════════════════════════════
    print("🗺️  Step 1: Phát hiện cụm địa lý tự động...")
    clusters = cluster_attractions(attractions, eps_km=18.0)
    for c in clusters:
        print(f"   Cụm {c.id}: {c.label} — {len(c.attractions)} điểm "
              f"({c.total_visit_minutes} phút thăm quan)")

    # ══════════════════════════════════════════════════════════════════
    # STEP 2 — ROUTE: Nối cụm thành tuyến 1 chiều
    # ══════════════════════════════════════════════════════════════════
    print("\n🛣️  Step 2: Sắp xếp tuyến đường tối ưu (Nearest Neighbor)...")
    ordered = order_clusters(clusters, start)
    route_str = " → ".join(c.label for c in ordered)
    print(f"\n   ✅ Tuyến cuối: {route_str}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 3 — ASSIGN: Chia cụm vào từng ngày
    # ══════════════════════════════════════════════════════════════════
    print(f"\n📅 Step 3: Phân bổ cụm vào {total_days} ngày "
          f"(budget {FULL_DAY_BUDGET}ph/ngày, overhead tính động theo khoảng cách thực)...")
    day_plans = assign_clusters_to_days(ordered, total_days, start)
    print("\n   Kết quả phân bổ:")
    for dp in day_plans:
        labels = [c.label for c in dp.clusters]
        print(f"   Ngày {dp.day_number}: {', '.join(labels) if labels else '(trống)'}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — DETAIL: Lên chi tiết từng ngày
    # ══════════════════════════════════════════════════════════════════
    print("\n✏️  Step 4: Lên lịch chi tiết từng ngày...")
    scheduler = DetailScheduler(food_places, accommodations, user_prefs)
    itinerary = []

    for dp in day_plans:
        cluster_labels = [c.label for c in dp.clusters]
        print(f"\n   {'─'*60}")
        print(f"   Ngày {dp.day_number} — {', '.join(cluster_labels) if cluster_labels else 'trống'}")
        print(f"   Pool: {sum(len(c.attractions) for c in dp.clusters)} điểm "
              f"từ {len(dp.clusters)} cụm")
        day_sched = scheduler.build_day_schedule(dp, total_days, start_date)
        itinerary.append(asdict(day_sched))
        print(f"   → Chọn được {day_sched.summary['total_attractions']} attractions")

    # ── Output ──────────────────────────────────────────────────────
    return {
        'destination': dest['name'],
        'user_preferences': user_prefs,
        'route_analysis': {
            'total_clusters': len(clusters),
            'cluster_order': [c.label for c in ordered],
        },
        'itinerary': itinerary,
        'meta': {
            'total_days': total_days,
            'total_attractions': sum(d['summary']['total_attractions'] for d in itinerary),
            'total_distance_km': sum(d['summary']['total_distance_km'] for d in itinerary)
        }
    }


# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_itinerary(result: Dict):
    meta = result['meta']
    route = result['route_analysis']

    print("\n" + "=" * 80)
    print(f"📋 LỊCH TRÌNH {meta['total_days']} NGÀY — {result['destination'].upper()}")
    print(f"   Tuyến: {' → '.join(route['cluster_order'])}")
    print("=" * 80)

    for day in result['itinerary']:
        print(f"\n{'━' * 80}")
        print(f"📅 NGÀY {day['day']}: {day['title']}  ({day['date']})")
        print(f"   Cụm: {', '.join(day['summary']['clusters_visited'])}")
        print(f"{'━' * 80}")

        for item in day['schedule']:
            t = item['time']

            if item['type'] == 'attraction':
                p = item['place']
                print(f"\n⏰ {t} — 📸 {p['name']}")
                print(f"   ⏱  {item['duration_minutes']} phút")
                if item.get('travel_time_minutes'):
                    print(f"   🚗 Di chuyển: {item['travel_time_minutes']} phút "
                          f"({item.get('distance_km', 0):.1f} km)")
                if p.get('tips'):
                    print(f"   💡 {p['tips'][0]}")
                for w in p.get('warnings') or []:
                    print(f"   ⚠️  {w.get('content', w)}")

            elif item['type'] == 'food':
                p = item['place']
                emoji = {'breakfast': '🍳', 'lunch': '🍜', 'dinner': '🍽️'}
                print(f"\n⏰ {t} — {emoji.get(item['meal_type'], '🍴')} "
                      f"{p['name']} ({item['meal_type'].title()})")
                print(f"   ⏱  {item['duration_minutes']} phút")
                if p.get('avg_spending'):
                    print(f"   💰 ~{p['avg_spending']:,} VNĐ/người")

            elif item['type'] == 'accommodation':
                p = item['place']
                print(f"\n⏰ {t} — 🏨 {p['name']}")
                if item.get('notes'):
                    print(f"   📍 {item['notes']}")

        s = day['summary']
        print(f"\n{'─' * 80}")
        print(f"📊 Tổng kết ngày {day['day']}:")
        print(f"   • Điểm tham quan: {s['total_attractions']}")
        print(f"   • Quãng đường: {s['total_distance_km']} km")
        elapsed = s['total_time_minutes']
        print(f"   • Thời gian hoạt động: {elapsed // 60}h {elapsed % 60}p")
        if s['accommodation']:
            print(f"   • Nghỉ đêm: {s['accommodation']}")

    print(f"\n{'=' * 80}")
    print("🎯 TỔNG KẾT CHUYẾN ĐI")
    print("=" * 80)
    print(f"   • Số ngày         : {meta['total_days']}")
    print(f"   • Tổng attractions: {meta['total_attractions']}")
    print(f"   • Tổng quãng đường: {meta['total_distance_km']:.1f} km")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("📂 Loading places data...")
    with open('route/ha_giang_places.json', 'r', encoding='utf-8') as f:
        places_data = json.load(f)

    print(f"   Attractions : {len(places_data['places']['attractions'])}")
    print(f"   Food        : {len(places_data['places']['food'])}")
    print(f"   Accommodations: {len(places_data['places']['accommodations'])}")

    user_prefs = {
        'days': 3,
        'interests': ['photography', 'healing'],
        'companions': 'couple',
        'budget': 'moderate',
        'start_date': '2024-03-15'
    }

    print(f"\n👤 User: {user_prefs['companions']}, "
          f"{user_prefs['days']} ngày, "
          f"interests: {', '.join(user_prefs['interests'])}")
    print()

    result = generate_itinerary(places_data, user_prefs)
    print_itinerary(result)

    out = 'generated_itinerary_v2.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved → {out}")


if __name__ == "__main__":
    main()