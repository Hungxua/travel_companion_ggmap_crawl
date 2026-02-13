from playwright.sync_api import sync_playwright, Page, Locator
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict, field
import json


SCROLLING = 30

@dataclass
class Review:
    rating: Optional[int] = None
    time_review: str = "UNKNOWN"
    content: str = "UNKNOWN"
    user_name: str = "UNKNOWN"
    user_level: str = "UNKNOWN"
    user_reviews: str = "UNKNOWN"
    user_photos: str = "UNKNOWN"
    visited_on: str = "UNKNOWN"
    wait_time: str = "UNKNOWN"
    reservation_recommended: str = "UNKNOWN"

@dataclass
class BusyHour:
    hour: int            # 8, 10, 11, 13
    percent: int         # 63, 52, 47


@dataclass
class BusyDay:
    day: str             # monday, tuesday...
    hours: List[BusyHour]


@dataclass
class BusyTime:
    days: Dict[str, BusyDay]


@dataclass
class Place:
    title: str = "UNKNOWN"
    stars: str = "UNKNOWN"
    num_reviews: str = "UNKNOWN"
    category: str = "UNKNOWN"
    address: str = "UNKNOWN"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    opening_time: List[Tuple[str, str]] = field(default_factory=list)
    contact_url: str = "UNKNOWN"
    phone: str = "UNKNOWN"
    reviews: List[Dict] = field(default_factory=list)
    busy_data: BusyTime = field(default_factory=dict)
    about: Dict = field(default_factory=dict)


class GoogleMapsScraper:
    def __init__(self, headless: bool = False, slow_mo: int = 50):
        self.headless = headless
        self.slow_mo = slow_mo
        self.page: Optional[Page] = None

    @staticmethod
    def parse_reviews(aria: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Parse review text to extract stars and review count"""
        if not aria:
            return None, None
        
        # "4.2 stars 1,234 Reviews" hoặc "4.2 sao 1,234 bài đánh giá"
        m = re.search(r"([\d.,]+)\s*(?:stars?|sao)\s*([\d.,]+)", aria)
        if not m:
            return None, None
        return m.group(1), m.group(2).replace(",", "")
    
    @staticmethod
    def parse_location(url: str) -> Tuple[Optional[float], Optional[float]]:
        m = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", url)
        if m:
            lat = float(m.group(1))
            lng = float(m.group(2))
            return lat, lng
        return None, None
    
    @staticmethod
    def parse_busy_label(label: str) -> BusyHour:
        percent_match = re.search(r"(\d+)\s*%", label)
        hour_match = re.search(r"lúc\s*(\d{1,2})|at\s*(\d{1,2})", label, re.IGNORECASE)

        if not percent_match or not hour_match:
            raise ValueError(f"Cannot parse busy label: {label}")

        percent = int(percent_match.group(1))
        hour = int(hour_match.group(1) or hour_match.group(2))

        return BusyHour(hour=hour, percent=percent)

    def safe_get_text(self, selector: str, timeout: int = 2000, default: str = "UNKNOWN") -> str:
        """Safely get text from element"""
        try:
            element = self.page.wait_for_selector(selector, timeout=timeout)
            text = element.inner_text() if element else default
            return text.strip() if text and text.strip() else default
        except Exception:
            return default

    def safe_get_attribute(self, selector: str, attribute: str, timeout: int = 2000, default: str = "UNKNOWN") -> str:
        """Safely get attribute from element"""
        try:
            element = self.page.wait_for_selector(selector, timeout=timeout)
            attr = element.get_attribute(attribute) if element else default
            return attr if attr else default
        except Exception:
            return default

    def extract_place_info(self) -> Place:
        """Extract all place information from the current page"""
        place = Place()

        # Location from URL
        current_url = self.page.url
        place.latitude, place.longitude = GoogleMapsScraper.parse_location(current_url)

        # Title
        place.title = self.safe_get_text("h1.DUwDvf")

        # Stars
        place.stars = self.safe_get_text("div.F7nice span[aria-hidden='true']")

        try:
            # 1. Click button mở menu
            busy_data = BusyTime(days={})
            for i in range(7):
                button = self.page.locator(
                    "button.e2moi[aria-haspopup='menu'][jsaction*='wfvdle']"
                )
                button.nth(2).click()
                
                self.page.wait_for_timeout(500)  # chờ menu render

                # 2. Lấy tất cả menu items
                self.page.locator("div[role='menuitemradio']").nth(i).click()
                self.page.wait_for_timeout(1000)
                busy_times = self.page.locator('div[role="img"][class="dpoVLd "]')
                count = busy_times.count()
                hours = []
                for j in range(count):
                    busy_time = busy_times.nth(j)
                    label = busy_time.get_attribute("aria-label")
                    busy_hour = GoogleMapsScraper.parse_busy_label(label)
                    if busy_hour.hour not in hours:
                        hours.append(busy_hour.hour)
                        day = self.page.locator("span[class='uEubGf NlVald']").nth(6).inner_text()
                        if day not in busy_data.days:
                            busy_data.days[day] = BusyDay(day=day, hours=[])

                        busy_data.days[day].hours.append(busy_hour)
            place.busy_data = busy_data
        except Exception as e:
            print(f'[extract] could not extract popular time: {e}')
            place.busy_data = {}

        # Reviews count
        try:
            reviews_node = self.page.query_selector(
                "span[role='img'][aria-label*='reviews'], span[role='img'][aria-label*='bài đánh giá']"
            )
            if reviews_node:
                reviews_text = reviews_node.get_attribute("aria-label")
                _, num_reviews = self.parse_reviews(reviews_text)
                place.num_reviews = num_reviews if num_reviews else "UNKNOWN"
            else:
                place.num_reviews = "UNKNOWN"
        except Exception as e:
            print(f"[extract] could not extract reviews count: {e}")
            place.num_reviews = "UNKNOWN"

        # Category
        place.category = self.safe_get_text("button[jsaction*='category']")

        # Address
        place.address = self.safe_get_text("button[data-item-id='address'] div.Io6YTe")

        # Opening hours
        place.opening_time = self.extract_opening_hours()

        # Contact URL
        place.contact_url = self.safe_get_attribute("a.CsEnBe", "href")

        # Phone
        place.phone = self.safe_get_text("button[data-item-id^='phone:tel'] div.Io6YTe")

        # Reviews
        # place.reviews = self.extract_reviews()

        # About
        place.about = self.extract_about()

        print(f"[extract] Done: {place.title} | stars={place.stars} | reviews={place.num_reviews} | lat={place.latitude} | lng={place.longitude} | about_keys={list(place.about.keys()) if isinstance(place.about, dict) else 'N/A'}")
        return place

    def extract_about(self) -> Dict:
        """Extract about"""
        about = {}
        try:
            # Tìm tab About động bằng text thay vì hardcode index
            all_tabs = self.page.locator("button[data-tab-index]")
            tab_count = all_tabs.count()

            about_node = None
            for t in range(tab_count):
                tab = all_tabs.nth(t)
                tab_label = tab.get_attribute("aria-label") or ""
                tab_text = tab.inner_text().strip()
                if tab_text.lower() in ("about", "giới thiệu") or "about" in tab_label.lower() or "giới thiệu" in tab_label.lower():
                    about_node = tab
                    break

            if about_node:
                about_node.click()
                self.page.wait_for_timeout(2000)

                sections = self.page.locator("div.iP2t7d")
                sections_count = sections.count()

                for i in range(sections_count):
                    section = sections.nth(i)
                    header = section.locator("h2.iL3Qke")
                    if header.count() > 0:
                        section_key = header.inner_text().strip().lower().replace(" ", "_")
                        items = []
                        li_tags = section.locator("li.hpLkke")
                        for j in range(li_tags.count()):
                            li = li_tags.nth(j)
                            label_span = li.locator("span[aria-label]")
                            if label_span.count() > 0:
                                items.append(label_span.get_attribute("aria-label"))
                            else:
                                items.append(li.inner_text().strip())
                        about[section_key] = items
            else:
                print(f"[extract_about] About tab not found (checked {tab_count} tabs)")

        except Exception as e:
            print(f"[extract_about] error: {e}")
            about = {}
        return about

    def extract_opening_hours(self) -> List[Tuple[str, str]]:
        """Extract opening hours"""
        opening_time = []
        try:
            rows = self.page.locator("tr.y0skZc")
            count = rows.count()

            for i in range(count):
                try:
                    row = rows.nth(i)
                    day = row.locator("td.ylH6lf").inner_text(timeout=1000)
                    time = row.locator("td.mxowUb li").inner_text(timeout=1000)
                    
                    day = day.strip() if day and day.strip() else "UNKNOWN"
                    time = time.strip() if time and time.strip() else "UNKNOWN"
                    
                    opening_time.append((day, time))
                except Exception:
                    opening_time.append(("UNKNOWN", "UNKNOWN"))
        except Exception as e:
            print(f"[extract] could not find opening hours: {e}")

        return opening_time

    def extract_reviews(self) -> List[Dict]:
        """Extract reviews from the place"""
        reviews = []
        try:
            # Click review button
            review_button = self.page.locator("button[jslog*='145620']")
            review_button.click(timeout=2000)
            self.page.wait_for_timeout(2000)

            # scroll 5 times
            # scroll cho thẻ div có class="m6QErb DxyBCb kA9KIf dS8AEf XiKgde", tabindex="-1", jslog="26354;mutable:true;"
            for _ in range(5):
                print('scrolling....')
                self.page.locator("div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde[tabindex='-1'][jslog='26354;mutable:true;']").evaluate("el => el.scrollTop += 1000")
                self.page.wait_for_timeout(2000)

            review_cards = self.page.locator("div.jftiEf")
            count = review_cards.count()

            for i in range(count):
                try:
                    review_card = review_cards.nth(i)
                    review = Review()

                    try:
                        # <button class="al6Kxe" jsaction="pane.wfvdle688.review.reviewerLink" data-review-id="Ci9DQUlRQUNvZENodHljRjlvT21zM1RIVTJYMVpQWVZOZlQwbGZVWGRVZVhZdFIwRRAB" data-href="https://www.google.com/maps/contrib/114428536243104399495/reviews?hl=en" jslog="95010; track:click;"><div class="d4r55 fontTitleMedium">Tia</div><div class="RfnDt ">Local Guide · 42 reviews · 56 photos</div></button>
                        # get username, level, number of reviews, number of photos
                        user_info = review_card.locator("button.al6Kxe")
                        review.user_name = user_info.locator("div.d4r55").inner_text(timeout=1000)
                        review.user_level = user_info.locator("div.RfnDt").inner_text(timeout=1000)
                    except Exception:
                        review.user_name = "UNKNOWN"
                        review.user_level = "UNKNOWN"

                    # Rating
                    try:
                        star_label = review_card.locator("span.kvMYJc[role='img']").get_attribute(
                            "aria-label", timeout=1000
                        )
                        if star_label:
                            rating_str = star_label.split()[0]
                            try:
                                review.rating = int(rating_str)
                            except (ValueError, IndexError):
                                review.rating = None
                        else:
                            review.rating = None
                    except Exception:
                        review.rating = None

                    # Time
                    try:
                        time_text = review_card.locator("span.rsqaWe").inner_text(timeout=1000)
                        review.time_review = time_text.strip() if time_text and time_text.strip() else "UNKNOWN"
                    except Exception:
                        review.time_review = "UNKNOWN"

                    # Content
                    try:
                        # <button class="w8nwRe kyuRq" aria-expanded="false" aria-controls="ChdDSUhNMG9nS0VJQ0FnSUNNOFpLM29nRRAB" data-review-id="ChdDSUhNMG9nS0VJQ0FnSUNNOFpLM29nRRAB" jslog="63707; track:click;metadata:WyIwYWhVS0V3aUZfYUw5cGMtU0F4V2c4WXNCSGR5WEZLd1EwcE1GQ0RJb0FnIl0=" aria-label="See more" jsaction="pane.wfvdle584.review.expandReview">More</button>
                        # if button exist, click it
                        more_button = review_card.locator("button.w8nwRe.kyuRq")
                        if more_button.count() > 0:
                            more_button.click(timeout=2000)
                            self.page.wait_for_timeout(2000)
                        content_text = review_card.locator("span.wiI7pd").inner_text(timeout=1000)
                        review.content = content_text.strip() if content_text and content_text.strip() else "UNKNOWN"
                        
                    except Exception:
                        review.content = "UNKNOWN"
                    
                    #  <div jslog="127691"><div jslog="126926;metadata:WyIwYWhVS0V3aTJtYUNncU0tU0F4VVZVRjRFSFRQVktXWVEzSWNIQ0lJQktBVSJd" class="PBK6be"><div><span class="RfDO5c"><span style="font-weight: bold;">Visited on</span></span></div><div><span class="RfDO5c"><span>Weekday</span><span jslog="126957;metadata:WyIwYWhVS0V3aTJtYUNncU0tU0F4VVZVRjRFSFRQVktXWVEzWWNIQ0lNQktBQSJd"></span></span></div></div><div jslog="126926;metadata:WyIwYWhVS0V3aTJtYUNncU0tU0F4VVZVRjRFSFRQVktXWVEzSWNIQ0lRQktBWSJd" class="PBK6be"><div><span class="RfDO5c"><span style="font-weight: bold;">Wait time</span></span></div><div><span class="RfDO5c"><span>No wait</span><span jslog="126957;metadata:WyIwYWhVS0V3aTJtYUNncU0tU0F4VVZVRjRFSFRQVktXWVEzWWNIQ0lVQktBQSJd"></span></span></div></div><div jslog="126926;metadata:WyIwYWhVS0V3aTJtYUNncU0tU0F4VVZVRjRFSFRQVktXWVEzSWNIQ0lZQktBYyJd" class="PBK6be"><div><span class="RfDO5c"><span style="font-weight: bold;">Reservation recommended</span></span></div><div><span class="RfDO5c"><span>No</span><span jslog="126957;metadata:WyIwYWhVS0V3aTJtYUNncU0tU0F4VVZVRjRFSFRQVktXWVEzWWNIQ0ljQktBQSJd"></span></span></div></div></div>
                    # get visited on, wait time, reservation recommended from div
                    try:
                        visited_on = review_card.locator("div.PBK6be").first.inner_text(timeout=1000)
                        wait_time = review_card.locator("div.PBK6be").nth(1).inner_text(timeout=1000)
                        reservation_recommended = review_card.locator("div.PBK6be").nth(2).inner_text(timeout=1000)
                        review.visited_on = visited_on.strip() if visited_on and visited_on.strip() else "UNKNOWN"
                        review.wait_time = wait_time.strip() if wait_time and wait_time.strip() else "UNKNOWN"
                        review.reservation_recommended = reservation_recommended.strip() if reservation_recommended and reservation_recommended.strip() else "UNKNOWN"
                    except Exception:
                        review.visited_on = "UNKNOWN"
                        review.wait_time = "UNKNOWN"
                        review.reservation_recommended = "UNKNOWN"
                    

                    reviews.append(asdict(review))

                except Exception:
                    reviews.append(asdict(Review()))

        except Exception as e:
            print(f"[extract] could not extract reviews: {e}")
        print('asdict(reviews)', reviews)
        return reviews

    def search_and_scrape(self, search_query: str) -> List[Dict]:
        """Main scraping function"""
        results = []

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
            )

            context = browser.new_context(
                viewport={"width": 1280, "height": 800},
                locale="vi-VN",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )

            self.page = context.new_page()

            try:
                # Open Google Maps
                print("Opening Google Maps...")
                self.page.goto("https://www.google.com/maps", timeout=60000)

                self.page.wait_for_timeout(3000)

                # Search
                print(f"Searching for: {search_query}")
                search_box = self.page.wait_for_selector('input[name="q"]')
                search_box.fill(search_query)
                self.page.keyboard.press("Enter")
                self.page.wait_for_timeout(5000)

                print("Search completed")

                # Click "Things to do" button
                things_btn = self.page.locator("button[jslog*='120706']")
                things_btn.first.click()
                self.page.wait_for_timeout(3000)

                feed = self.page.locator("div[role='feed']")
                for i in range(SCROLLING):
                    print(f'Scrolling... {i}')
                    feed.hover()
                    self.page.mouse.wheel(0, 3000)
                    self.page.wait_for_timeout(2000)

                # Get all cards
                cards = self.page.locator("div[role='article']")
                count = cards.count()
                print(f"Found {count} places")

                # Process each card
                for i in range(count):
                    print(f"\n{'='*60} Place {i + 1}/{count} {'='*60}")

                    try:
                        card = cards.nth(i)
                        card.click()
                        self.page.wait_for_timeout(3000)

                        # Extract place info (location is extracted inside)
                        place = self.extract_place_info()
                        results.append(asdict(place))

                    except Exception as e:
                        print(f"Error processing card {i}: {e}")
                        results.append(asdict(Place()))
                        continue

            except Exception as e:
                print(f"Fatal error during scraping: {e}")

            finally:
                self.page.wait_for_timeout(3000)
                browser.close()

        return results


def main():
    scraper = GoogleMapsScraper(headless=False, slow_mo=50)
    results = scraper.search_and_scrape("Hà Giang")

    # Print results
    print("\n" + "=" * 80)
    print(f"FINAL RESULTS: {len(results)} places")
    print("=" * 80)
    
    for i, place in enumerate(results, 1):
        print(f"\n{i}. {place['title']}")
        print(f"   Stars: {place['stars']} ({place['num_reviews']} reviews)")
        print(f"   Category: {place['category']}")
        print(f"   Address: {place['address']}")
        print(f"   Location: ({place['latitude']}, {place['longitude']})")
        print(f"   Phone: {place['phone']}")
        print(f"   Website: {place['contact_url']}")
        print(f"   Reviews: {len(place['reviews'])} extracted")
        print(f"   About: {place['about']}")

    # Save to JSON
    with open("google_maps_accomodations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(results)} places to google_maps_accomodations.json")


if __name__ == "__main__":
    main()