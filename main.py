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
    opening_time: List[Tuple[str, str]] = field(default_factory=list)
    contact_url: str = "UNKNOWN"
    phone: str = "UNKNOWN"
    reviews: List[Dict] = field(default_factory=list)
    busy_data: BusyTime = field(default_factory=dict)


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
        except Exception as e:
            print(f"Could not find element with selector '{selector}': {e}")
            return default

    def safe_get_attribute(self, selector: str, attribute: str, timeout: int = 2000, default: str = "UNKNOWN") -> str:
        """Safely get attribute from element"""
        try:
            element = self.page.wait_for_selector(selector, timeout=timeout)
            attr = element.get_attribute(attribute) if element else default
            return attr if attr else default
        except Exception as e:
            print(f"Could not get attribute '{attribute}' from '{selector}': {e}")
            return default

    def extract_place_info(self) -> Place:
        """Extract all place information from the current page"""
        place = Place()

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
                # print('button', button)
                button.nth(2).click()
                
                self.page.wait_for_timeout(500)  # chờ menu render

                # 2. Lấy tất cả menu items
                # items = self.page.locator("div[role='menuitemradio']")
                self.page.locator("div[role='menuitemradio']").nth(i).click()
                self.page.wait_for_timeout(1000)
                busy_times = self.page.locator('div[role="img"][class="dpoVLd "]')
                count = busy_times.count()
                for j in range(count):
                    busy_time = busy_times.nth(j)
                    label = busy_time.get_attribute("aria-label")
                    busy_hour = GoogleMapsScraper.parse_busy_label(label)
                    day = self.page.locator("span[class='uEubGf NlVald']").nth(6).inner_text()
                    if day not in busy_data.days:
                        busy_data.days[day] = BusyDay(day=day, hours=[])

                    busy_data.days[day].hours.append(busy_hour)
            place.busy_data = busy_data
        except Exception as e:
            print(f'could not extract popular time: {e}')
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
            print(f"Could not extract reviews count: {e}")
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
        place.reviews = self.extract_reviews()

        return place

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
                except Exception as e:
                    print(f"Could not extract opening hour for row {i}: {e}")
                    opening_time.append(("UNKNOWN", "UNKNOWN"))
        except Exception as e:
            print(f"Could not find opening hours table: {e}")

        return opening_time

    def extract_reviews(self) -> List[Dict]:
        """Extract reviews from the place"""
        reviews = []
        try:
            # Click review button
            review_button = self.page.locator("button[jslog*='145620']")
            review_button.click(timeout=2000)
            self.page.wait_for_timeout(2000)

            review_cards = self.page.locator("div.jftiEf")
            count = review_cards.count()

            for i in range(count):
                try:
                    review_card = review_cards.nth(i)
                    review = Review()

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
                    except Exception as e:
                        print(f"Could not extract rating for review {i}: {e}")
                        review.rating = None

                    # Time
                    try:
                        time_text = review_card.locator("span.rsqaWe").inner_text(timeout=1000)
                        review.time_review = time_text.strip() if time_text and time_text.strip() else "UNKNOWN"
                    except Exception as e:
                        print(f"Could not extract time for review {i}: {e}")
                        review.time_review = "UNKNOWN"

                    # Content
                    try:
                        content_text = review_card.locator("span.wiI7pd").inner_text(timeout=1000)
                        review.content = content_text.strip() if content_text and content_text.strip() else "UNKNOWN"
                    except Exception as e:
                        print(f"Could not extract content for review {i}: {e}")
                        review.content = "UNKNOWN"

                    reviews.append(asdict(review))
                    print(f"Review {i}: rating={review.rating}, time={review.time_review}")

                except Exception as e:
                    print(f"Error extracting review {i}: {e}")
                    # Add a placeholder review
                    reviews.append(asdict(Review()))

        except Exception as e:
            print(f"Could not extract reviews: {e}")

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

                # Search
                print(f"Searching for: {search_query}")
                search_box = self.page.wait_for_selector('input[name="q"]')
                search_box.fill(search_query)
                self.page.keyboard.press("Enter")
                self.page.wait_for_timeout(2000)

                print("Search completed")

                # Click "Things to do" button
                things_btn = self.page.locator("button[jslog*='150577']")
                things_btn.first.click()
                self.page.wait_for_timeout(3000)

                feed = self.page.locator("div[role='feed']")
                for i in range(SCROLLING):
                    print('scrolling......', i)
                    feed.hover()
                    self.page.mouse.wheel(0, 3000)
                    self.page.wait_for_timeout(2000)

                # Get all cards
                cards = self.page.locator("div[role='article']")
                count = cards.count()
                print(f"Found {count} places")

                # Process each card
                for i in range(count):
                    print('=' * 100)
                    print(f"Processing place {i + 1}/{count}")
                    print('=' * 100)

                    try:
                        card = cards.nth(i)
                        card.click()
                        self.page.wait_for_timeout(3000)

                        # Extract place info
                        place = self.extract_place_info()
                        results.append(asdict(place))

                        print(f"Extracted: {place.title}")

                    except Exception as e:
                        print(f"Error processing card {i}: {e}")
                        # Add placeholder place
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
    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    
    for i, place in enumerate(results, 1):
        print(f"\n{i}. {place['title']}")
        print(f"   Stars: {place['stars']} ({place['num_reviews']} reviews)")
        print(f"   Category: {place['category']}")
        print(f"   Address: {place['address']}")
        print(f"   Phone: {place['phone']}")
        print(f"   Website: {place['contact_url']}")
        print(f"   Reviews: {len(place['reviews'])} extracted")

    # Save to JSON
    with open("google_maps_results1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(results)} places to google_maps_results1.json")


if __name__ == "__main__":
    main()