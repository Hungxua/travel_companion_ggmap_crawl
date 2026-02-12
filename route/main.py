"""
COMPLETE ITINERARY GENERATOR
Tạo lịch trình du lịch tự động với greedy algorithm

Features:
- Read places from JSON file
- Greedy selection for attractions
- Rule-based meal insertion
- Smart accommodation placement (20km threshold)
- Time constraint handling
- Complete day-by-day schedule output
"""

import json
import math
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# DATA MODELS
# ============================================================================

class PlaceType(Enum):
    ATTRACTION = "attraction"
    FOOD = "food"
    ACCOMMODATION = "accommodation"


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
class ScheduleItem:
    time: str
    type: str  # 'attraction', 'food', 'accommodation', 'transport'
    place: Optional[Dict] = None
    duration_minutes: Optional[int] = None
    travel_time_minutes: Optional[int] = None
    distance_km: Optional[float] = None
    meal_type: Optional[str] = None
    activity: Optional[str] = None
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
    """
    Tính khoảng cách giữa 2 điểm theo công thức Haversine
    Returns: distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lat1_rad = math.radians(coord1.lat)
    lat2_rad = math.radians(coord2.lat)
    delta_lat = math.radians(coord2.lat - coord1.lat)
    delta_lng = math.radians(coord2.lng - coord1.lng)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lng / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def estimate_travel_time(coord1: Coordinates, coord2: Coordinates, 
                         is_mountain: bool = True) -> int:
    """
    Ước tính thời gian di chuyển giữa 2 điểm
    Returns: time in minutes
    """
    distance_km = haversine_distance(coord1, coord2)
    
    # Mountain roads are slower and winding
    if is_mountain:
        # Actual road distance = straight line × 1.3-1.5
        actual_distance = distance_km * 1.4
        # Average speed: 25 km/h on mountain roads
        speed = 25
    else:
        actual_distance = distance_km * 1.1
        speed = 40
    
    time_hours = actual_distance / speed
    return int(time_hours * 60)


def parse_time(time_str: str) -> datetime:
    """Parse time string to datetime"""
    return datetime.strptime(time_str, "%H:%M")


def format_time(dt: datetime) -> str:
    """Format datetime to time string"""
    return dt.strftime("%H:%M")


def add_minutes(dt: datetime, minutes: int) -> datetime:
    """Add minutes to datetime"""
    return dt + timedelta(minutes=minutes)


# ============================================================================
# ITINERARY GENERATOR CLASS
# ============================================================================

class ItineraryGenerator:
    """
    Main class for generating travel itinerary using greedy algorithm
    """
    
    # Constants
    MAX_DAILY_TIME_MINUTES = 480  # 8 hours
    MAX_ATTRACTIONS_PER_DAY = 4
    DAILY_START_TIME = "08:00"
    ACCOMMODATION_DISTANCE_THRESHOLD = 20  # km
    
    def __init__(self, places_data: Dict, user_preferences: Dict):
        """
        Args:
            places_data: Dict with 'attractions', 'food', 'accommodations'
            user_preferences: {
                'days': 3,
                'interests': ['healing', 'photography'],
                'companions': 'couple',
                'budget': 'moderate',
                'start_date': '2024-03-15'
            }
        """
        self.destination = places_data['destination']
        self.attractions = [self._dict_to_place(p) for p in places_data['places']['attractions']]
        self.food_places = [self._dict_to_place(p) for p in places_data['places']['food']]
        self.accommodations = [self._dict_to_place(p) for p in places_data['places']['accommodations']]
        
        self.user_prefs = user_preferences
        self.visited_place_ids = set()
        
        # Start location
        self.start_location = Coordinates(
            lat=self.destination['start_location']['lat'],
            lng=self.destination['start_location']['lng']
        )
    
    def _dict_to_place(self, data: Dict) -> Place:
        """Convert dict to Place object"""
        coords = Coordinates(
            lat=data['coordinates']['lat'],
            lng=data['coordinates']['lng']
        )
        
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
    
    # ========================================================================
    # SCORING FUNCTIONS
    # ========================================================================
    
    def calculate_place_score(self, place: Place) -> float:
        """
        Tính điểm cho một địa điểm dựa trên user preferences
        """
        interests = self.user_prefs.get('interests', [])
        companion = self.user_prefs.get('companions', 'solo')
        
        # 1. Interest matching score
        interest_score = 0
        for interest in interests:
            if interest in place.vibe_scores:
                interest_score += place.vibe_scores[interest] * 3.0
        
        # Add other vibes with lower weight
        for vibe, score in place.vibe_scores.items():
            if vibe not in interests:
                interest_score += score * 0.5
        
        # 2. Companion score
        companion_bonus = place.companion_scores.get(companion, 0.5)
        
        # 3. Priority score
        priority_bonus = place.priority_score * 0.5
        
        # 4. Must-visit bonus
        must_visit_bonus = 2.0 if place.must_visit else 0
        
        # 5. Popularity score (from reviews)
        rating_normalized = place.google_maps_rating / 5.0
        review_log = math.log10(place.google_maps_reviews_count + 1)
        popularity = rating_normalized * review_log
        
        # Combined score
        final_score = (
            interest_score * 0.4 +
            companion_bonus * 0.2 +
            priority_bonus * 0.1 +
            popularity * 0.2 +
            must_visit_bonus * 0.1
        )
        
        return final_score
    
    def calculate_candidate_score(self, place: Place, current_location: Coordinates,
                                  time_remaining: int) -> Optional[Dict]:
        """
        Tính điểm cho candidate place trong greedy selection
        """
        # Check if already visited
        if place.id in self.visited_place_ids:
            return None
        
        # Calculate travel time and distance
        travel_time = estimate_travel_time(current_location, place.coordinates)
        distance_km = haversine_distance(current_location, place.coordinates)
        total_time = travel_time + place.visit_duration_minutes
        
        # Check time constraint
        if total_time > time_remaining:
            return None
        
        # Place quality score
        quality_score = self.calculate_place_score(place)
        
        # Distance penalty (farther = lower score)
        distance_penalty = distance_km / 100  # Normalize
        
        # Time penalty
        time_penalty = total_time / 240  # Normalize to 4 hours
        
        # Must-visit bonus
        must_visit_bonus = 2.0 if place.must_visit else 0
        
        # Combined score
        combined_score = (
            quality_score * 0.5 +
            (1 - distance_penalty) * 0.2 +
            (1 - time_penalty) * 0.2 +
            must_visit_bonus * 0.1
        )
        
        return {
            'place': place,
            'distance_km': distance_km,
            'travel_time_minutes': travel_time,
            'total_time_minutes': total_time,
            'quality_score': quality_score,
            'combined_score': combined_score
        }
    
    # ========================================================================
    # GREEDY SELECTION
    # ========================================================================
    
    def greedy_select_attractions(self, start_location: Coordinates,
                                  time_budget: int, max_count: int = 4) -> List[Dict]:
        """
        Greedy algorithm để chọn attractions
        """
        selected = []
        current_location = start_location
        time_remaining = time_budget
        
        while len(selected) < max_count and time_remaining > 60:
            # Get all candidates
            candidates = []
            for attraction in self.attractions:
                candidate = self.calculate_candidate_score(
                    attraction,
                    current_location,
                    time_remaining
                )
                if candidate:
                    candidates.append(candidate)
            
            if not candidates:
                break
            
            # Sort by combined score
            candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Select best candidate
            best = candidates[0]
            selected.append(best)
            
            # Update state
            self.visited_place_ids.add(best['place'].id)
            current_location = best['place'].coordinates
            time_remaining -= best['total_time_minutes']
        
        return selected
    
    # ========================================================================
    # ACCOMMODATION PLANNING
    # ========================================================================
    
    def determine_accommodation(self, day_start: Coordinates,
                               day_end: Coordinates, day_number: int,
                               total_days: int) -> Optional[Place]:
        """
        Xác định accommodation dựa trên quy tắc 20km
        """
        # Last day - no accommodation needed
        if day_number == total_days:
            return None
        
        # Calculate distance from start to end
        distance_km = haversine_distance(day_start, day_end)
        
        # Decide target location for accommodation
        if distance_km > self.ACCOMMODATION_DISTANCE_THRESHOLD:
            # Move forward - sleep near end point
            target_location = day_end
            reason = f"Di chuyển tới khu vực mới (đã xa {distance_km:.1f}km)"
        else:
            # Stay near start - return to start area
            target_location = day_start
            reason = f"Ở lại khu vực hiện tại (chỉ cách {distance_km:.1f}km)"
        
        # Find best accommodation near target
        accommodation = self.find_best_accommodation(target_location)
        
        if accommodation:
            return {
                'accommodation': accommodation,
                'decision_reason': reason,
                'start_to_end_distance': distance_km
            }
        
        return None
    
    def find_best_accommodation(self, target_location: Coordinates,
                               max_distance_km: float = 10) -> Optional[Place]:
        """
        Tìm accommodation tốt nhất gần target location
        """
        candidates = []
        
        for acc in self.accommodations:
            distance_km = haversine_distance(target_location, acc.coordinates)
            
            if distance_km > max_distance_km:
                continue
            
            # Scoring
            distance_score = 1 - (distance_km / max_distance_km)
            quality_score = acc.google_maps_rating / 5.0
            
            combined_score = (
                distance_score * 0.6 +
                quality_score * 0.4
            )
            
            candidates.append({
                'accommodation': acc,
                'distance_km': distance_km,
                'combined_score': combined_score
            })
        
        if not candidates:
            # Fallback: expand search radius
            return self.find_best_accommodation(target_location, max_distance_km * 2)
        
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        return candidates[0]['accommodation']
    
    # ========================================================================
    # MEAL PLANNING
    # ========================================================================
    
    def find_nearest_food(self, location: Coordinates, meal_type: str,
                         current_time: datetime) -> Optional[Place]:
        """
        Tìm địa điểm ăn uống gần nhất
        """
        # Filter by meal_type
        candidates = [
            f for f in self.food_places
            if f.meal_type in [meal_type, 'all_day']
        ]
        
        if not candidates:
            # Fallback to all_day places
            candidates = [f for f in self.food_places if f.meal_type == 'all_day']
        
        if not candidates:
            return None
        
        # Find nearest
        candidates_with_distance = [
            {
                'place': p,
                'distance': haversine_distance(location, p.coordinates)
            }
            for p in candidates
        ]
        
        candidates_with_distance.sort(key=lambda x: x['distance'])
        
        return candidates_with_distance[0]['place']
    
    # ========================================================================
    # SCHEDULE GENERATION
    # ========================================================================
    
    def generate_day_schedule(self, day_number: int, total_days: int,
                             start_location: Coordinates,
                             start_date: str) -> DaySchedule:
        """
        Generate schedule cho một ngày
        """
        schedule_items = []
        current_time = parse_time(self.DAILY_START_TIME)
        current_location = start_location
        
        # Calculate date
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        day_date = (start_dt + timedelta(days=day_number - 1)).strftime("%Y-%m-%d")
        
        # =====================================================
        # PHASE 1: Select attractions (Greedy)
        # =====================================================
        
        time_budget = self.MAX_DAILY_TIME_MINUTES
        
        # Reserve time for meals
        time_budget -= (45 + 60 + 60)  # breakfast + lunch + dinner
        
        selected_attractions = self.greedy_select_attractions(
            start_location,
            time_budget,
            self.MAX_ATTRACTIONS_PER_DAY
        )
        
        # Determine day end point
        if selected_attractions:
            day_end_location = selected_attractions[-1]['place'].coordinates
        else:
            day_end_location = start_location
        
        # =====================================================
        # PHASE 2: Plan accommodation
        # =====================================================
        
        accommodation_info = self.determine_accommodation(
            start_location,
            day_end_location,
            day_number,
            total_days
        )
        
        # =====================================================
        # PHASE 3: Build schedule with meals
        # =====================================================
        
        total_distance = 0
        
        # Breakfast (if not day 1)
        if day_number > 1:
            breakfast_place = self.find_nearest_food(
                current_location,
                'breakfast',
                current_time
            )
            
            if breakfast_place:
                # Travel to breakfast
                travel_time = estimate_travel_time(current_location, breakfast_place.coordinates)
                current_time = add_minutes(current_time, travel_time)
                
                schedule_items.append(ScheduleItem(
                    time=format_time(current_time),
                    type='food',
                    meal_type='breakfast',
                    place=self._place_to_dict(breakfast_place),
                    duration_minutes=45,
                    travel_time_minutes=travel_time
                ))
                
                current_time = add_minutes(current_time, 45)
                current_location = breakfast_place.coordinates
        
        # Process attractions with lunch insertion
        lunch_inserted = False
        
        for i, attr_info in enumerate(selected_attractions):
            attraction = attr_info['place']
            
            # Travel to attraction
            travel_time = attr_info['travel_time_minutes']
            travel_distance = attr_info['distance_km']
            current_time = add_minutes(current_time, travel_time)
            
            # Visit attraction
            schedule_items.append(ScheduleItem(
                time=format_time(current_time),
                type='attraction',
                place=self._place_to_dict(attraction),
                duration_minutes=attraction.visit_duration_minutes,
                travel_time_minutes=travel_time,
                distance_km=travel_distance
            ))
            
            current_time = add_minutes(current_time, attraction.visit_duration_minutes)
            current_location = attraction.coordinates
            total_distance += travel_distance
            
            # Check if need lunch (11:30-13:30 window)
            hour = current_time.hour
            if not lunch_inserted and 11 <= hour <= 14:
                lunch_place = self.find_nearest_food(
                    current_location,
                    'lunch',
                    current_time
                )
                
                if lunch_place:
                    lunch_travel_time = estimate_travel_time(
                        current_location,
                        lunch_place.coordinates
                    )
                    current_time = add_minutes(current_time, lunch_travel_time)
                    
                    schedule_items.append(ScheduleItem(
                        time=format_time(current_time),
                        type='food',
                        meal_type='lunch',
                        place=self._place_to_dict(lunch_place),
                        duration_minutes=60,
                        travel_time_minutes=lunch_travel_time
                    ))
                    
                    current_time = add_minutes(current_time, 60)
                    current_location = lunch_place.coordinates
                    lunch_inserted = True
        
        # Dinner
        if accommodation_info:
            acc_location = accommodation_info['accommodation'].coordinates
        else:
            acc_location = current_location
        
        dinner_place = self.find_nearest_food(
            acc_location,
            'dinner',
            current_time
        )
        
        if dinner_place:
            dinner_travel_time = estimate_travel_time(
                current_location,
                dinner_place.coordinates
            )
            current_time = add_minutes(current_time, dinner_travel_time)
            
            schedule_items.append(ScheduleItem(
                time=format_time(current_time),
                type='food',
                meal_type='dinner',
                place=self._place_to_dict(dinner_place),
                duration_minutes=60,
                travel_time_minutes=dinner_travel_time
            ))
            
            current_time = add_minutes(current_time, 60)
            current_location = dinner_place.coordinates
        
        # Accommodation
        if accommodation_info:
            acc_travel_time = estimate_travel_time(
                current_location,
                accommodation_info['accommodation'].coordinates
            )
            current_time = add_minutes(current_time, acc_travel_time)
            
            schedule_items.append(ScheduleItem(
                time=format_time(current_time),
                type='accommodation',
                place=self._place_to_dict(accommodation_info['accommodation']),
                notes=accommodation_info['decision_reason']
            ))
        
        # Build day title
        if day_number == 1:
            title = f"Hà Nội → {selected_attractions[0]['place'].district if selected_attractions else 'Hà Giang'}"
        elif day_number == total_days:
            title = f"{selected_attractions[0]['place'].district if selected_attractions else 'Hà Giang'} → Hà Nội"
        else:
            title = f"Khám phá {selected_attractions[0]['place'].district if selected_attractions else 'Hà Giang'}"
        
        # Summary
        summary = {
            'total_attractions': len(selected_attractions),
            'total_distance_km': round(total_distance, 1),
            'total_time_minutes': int((current_time - parse_time(self.DAILY_START_TIME)).total_seconds() / 60),
            'accommodation': accommodation_info['accommodation'].name if accommodation_info else None
        }
        
        return DaySchedule(
            day=day_number,
            date=day_date,
            title=title,
            schedule=[asdict(item) for item in schedule_items],
            summary=summary
        )
    
    def _place_to_dict(self, place: Place) -> Dict:
        """Convert Place to dict for JSON serialization"""
        return {
            'id': place.id,
            'name': place.name,
            'type': place.poi_type,
            'coordinates': {'lat': place.coordinates.lat, 'lng': place.coordinates.lng},
            'rating': place.google_maps_rating,
            'tips': place.tips,
            'warnings': place.warnings,
            'avg_spending': place.avg_spending
        }
    
    # ========================================================================
    # MAIN GENERATION
    # ========================================================================
    
    def generate(self) -> Dict:
        """
        Generate complete itinerary
        """
        days = self.user_prefs.get('days', 3)
        start_date = self.user_prefs.get('start_date', '2024-03-15')
        
        itinerary = []
        current_start_location = self.start_location
        
        for day_num in range(1, days + 1):
            day_schedule = self.generate_day_schedule(
                day_num,
                days,
                current_start_location,
                start_date
            )
            
            itinerary.append(asdict(day_schedule))
            
            # Update start location for next day
            # = accommodation location if exists
            if day_schedule.summary['accommodation']:
                # Find accommodation coordinates from schedule
                acc_item = next(
                    (item for item in day_schedule.schedule if item['type'] == 'accommodation'),
                    None
                )
                if acc_item and acc_item['place']:
                    current_start_location = Coordinates(
                        lat=acc_item['place']['coordinates']['lat'],
                        lng=acc_item['place']['coordinates']['lng']
                    )
        
        return {
            'destination': self.destination['name'],
            'user_preferences': self.user_prefs,
            'itinerary': itinerary,
            'meta': {
                'total_days': days,
                'total_attractions': sum(day['summary']['total_attractions'] for day in itinerary),
                'total_distance_km': sum(day['summary']['total_distance_km'] for day in itinerary)
            }
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    
    # 1. Load places data from JSON
    print("📂 Loading places data from JSON...")
    with open('route\ha_giang_places.json', 'r', encoding='utf-8') as f:
        places_data = json.load(f)
    
    print(f"✅ Loaded {len(places_data['places']['attractions'])} attractions")
    print(f"✅ Loaded {len(places_data['places']['food'])} food places")
    print(f"✅ Loaded {len(places_data['places']['accommodations'])} accommodations")
    print()
    
    # 2. Define user preferences
    user_preferences = {
        'days': 3,
        'interests': ['photography', 'healing'],
        'companions': 'couple',
        'budget': 'moderate',
        'start_date': '2024-03-15'
    }
    
    print("👤 User Preferences:")
    print(f"   Days: {user_preferences['days']}")
    print(f"   Interests: {', '.join(user_preferences['interests'])}")
    print(f"   Companions: {user_preferences['companions']}")
    print(f"   Budget: {user_preferences['budget']}")
    print(f"   Start date: {user_preferences['start_date']}")
    print()
    
    # 3. Generate itinerary
    print("🚀 Generating itinerary with Greedy Algorithm...")
    print()
    
    generator = ItineraryGenerator(places_data, user_preferences)
    itinerary = generator.generate()
    
    # 4. Print results
    print("=" * 80)
    print(f"📋 LỊCH TRÌNH {itinerary['meta']['total_days']} NGÀY - {itinerary['destination'].upper()}")
    print("=" * 80)
    print()
    
    for day in itinerary['itinerary']:
        print(f"{'━' * 80}")
        print(f"📅 NGÀY {day['day']}: {day['title']}")
        print(f"   Ngày: {day['date']}")
        print(f"{'━' * 80}")
        
        for item in day['schedule']:
            time_str = item['time']
            
            if item['type'] == 'attraction':
                place = item['place']
                print(f"\n⏰ {time_str} - 📸 {place['name']}")
                print(f"   ⏱  Thời gian: {item['duration_minutes']} phút")
                if item.get('travel_time_minutes'):
                    print(f"   🚗 Di chuyển: {item['travel_time_minutes']} phút ({item.get('distance_km', 0):.1f} km)")
                if place.get('tips'):
                    print(f"   💡 Tips: {place['tips'][0]}")
                if place.get('warnings'):
                    for warn in place['warnings']:
                        print(f"   ⚠️  {warn['content']}")
            
            elif item['type'] == 'food':
                place = item['place']
                meal_emoji = {'breakfast': '🍳', 'lunch': '🍜', 'dinner': '🍽️'}.get(item['meal_type'], '🍴')
                print(f"\n⏰ {time_str} - {meal_emoji} {place['name']} ({item['meal_type'].title()})")
                print(f"   ⏱  Thời gian: {item['duration_minutes']} phút")
                if place.get('avg_spending'):
                    print(f"   💰 Chi phí TB: {place['avg_spending']:,} VNĐ")
            
            elif item['type'] == 'accommodation':
                place = item['place']
                print(f"\n⏰ {time_str} - 🏨 {place['name']}")
                if item.get('notes'):
                    print(f"   📍 {item['notes']}")
        
        print(f"\n{'─' * 80}")
        print(f"📊 Tổng kết:")
        print(f"   • Điểm tham quan: {day['summary']['total_attractions']}")
        print(f"   • Quãng đường: {day['summary']['total_distance_km']:.1f} km")
        print(f"   • Thời gian hoạt động: {day['summary']['total_time_minutes'] // 60}h {day['summary']['total_time_minutes'] % 60}p")
        if day['summary']['accommodation']:
            print(f"   • Nghỉ đêm tại: {day['summary']['accommodation']}")
        print()
    
    print("=" * 80)
    print(f"🎯 TỔNG KẾT CHUYẾN ĐI")
    print("=" * 80)
    print(f"   • Tổng số ngày: {itinerary['meta']['total_days']}")
    print(f"   • Tổng điểm tham quan: {itinerary['meta']['total_attractions']}")
    print(f"   • Tổng quãng đường: {itinerary['meta']['total_distance_km']:.1f} km")
    print()
    
    # 5. Save to JSON file
    output_file = 'generated_itinerary.json'
    print(f"💾 Saving itinerary to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(itinerary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Done! Itinerary saved to {output_file}")
    print()


if __name__ == "__main__":
    main()