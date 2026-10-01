"""Generate the synthetic retail-sales CSVs in datasets/sales/raw/ (deterministic, seeded).

    uv run python scripts/make_sales_data.py            # write the 4 CSVs (overwrites)
    uv run python scripts/make_sales_data.py --seed 7   # a different but still repeatable set

The files look like a real Indian e-commerce export on purpose, so loading them exercises the
generic loader rather than a tidy fixture: Title-Case headers with spaces and brackets
("Price (INR)"), day-first dates (27/09/2024), grouped amounts ("1,299.00"), yes/no flags,
leading-zero pincodes and cities typed in inconsistent case. The loader must handle all of it
from dataset.yaml alone (zero code changes, ADR-009).
"""

import argparse
import csv
import random
import sys
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "sales" / "raw"

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Ishaan", "Arjun", "Rohan", "Kabir", "Ananya", "Diya", "Saanvi",
    "Meera", "Priya", "Neha", "Kavya", "Riya", "Vikram", "Rahul", "Sneha", "Pooja", "Karan",
]  # fmt: skip
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Iyer", "Reddy", "Nair", "Gupta", "Mehta", "Shah", "Rao",
    "Joshi", "Desai", "Kulkarni", "Singh", "Das",
]  # fmt: skip
# city -> (weight, pincode prefix). ~5 % of pincodes get a leading 0 on purpose (see generate()).
CITIES = {
    "Mumbai": (18, "400"), "Delhi": (16, "110"), "Bengaluru": (15, "560"),
    "Hyderabad": (10, "500"), "Chennai": (9, "600"), "Pune": (9, "411"),
    "Ahmedabad": (8, "380"), "Kolkata": (8, "700"), "Jaipur": (4, "302"), "Kochi": (3, "682"),
}  # fmt: skip
CATALOG = {  # category -> [(product name, list price INR)]
    "Electronics": [
        ("Wireless Earbuds", 1999), ("Bluetooth Speaker", 2499), ("Smartwatch", 4999),
        ("Power Bank 20000mAh", 1499), ("USB-C Charger 65W", 1299), ("Mechanical Keyboard", 3499),
        ("Wireless Mouse", 799), ("27in Monitor", 14999), ("Noise Cancelling Headphones", 8999),
        ("Fitness Band", 2199),
    ],
    "Apparel": [
        ("Cotton Kurta", 1199), ("Denim Jeans", 1799), ("Running Shoes", 2999), ("Hoodie", 1499),
        ("Linen Shirt", 1399), ("Saree", 3499), ("Sports T-Shirt", 599), ("Rain Jacket", 2299),
        ("Leather Belt", 699), ("Woollen Shawl", 1899),
    ],
    "Home": [
        ("Pressure Cooker 5L", 2199), ("Non-stick Tawa", 899), ("Bedsheet Set", 1299),
        ("Table Lamp", 1099), ("Water Purifier", 12999), ("Mixer Grinder", 3999),
        ("Steel Bottle Set", 749), ("Cushion Covers (5)", 599), ("Wall Clock", 849),
        ("Storage Boxes (3)", 999),
    ],
    "Books": [
        ("The Guide - R.K. Narayan", 299), ("Wings of Fire", 349), ("Clean Code", 899),
        ("Atomic Habits", 499), ("Malgudi Days", 249), ("Sapiens", 599),
        ("Designing Data-Intensive Applications", 1499), ("The White Tiger", 399),
        ("Ikigai", 350), ("Rich Dad Poor Dad", 399),
    ],
}  # fmt: skip
STATUSES = ["delivered", "shipped", "placed", "cancelled", "returned"]
STATUS_WEIGHTS = [70, 8, 4, 10, 8]
CHANNELS = {"web": 45, "app": 45, "phone": 10}

START, END = date(2023, 1, 1), date(2025, 12, 31)
N_CUSTOMERS, N_ORDERS = 150, 420


def generate(seed: int) -> dict[str, list[list[str]]]:
    rnd = random.Random(seed)
    tables: dict[str, list[list[str]]] = {}

    customers = [["Customer ID", "Name", "City", "Pincode", "Signup Date", "Is Premium"]]
    signups: dict[int, date] = {}
    for cid in range(1, N_CUSTOMERS + 1):
        city = rnd.choices(list(CITIES), weights=[w for w, _ in CITIES.values()])[0]
        pincode = CITIES[city][1] + f"{rnd.randint(1, 99):03d}"
        if rnd.random() < 0.05:
            pincode = "0" + pincode[1:]  # a leading-zero code must survive as text
        typed_city = city
        roll = rnd.random()
        if roll < 0.08:
            typed_city = city.lower()  # customers type their own city
        elif roll < 0.11:
            typed_city = city.upper()
        elif roll < 0.13:
            typed_city = ""  # left blank -> NULL
        signup = START - timedelta(days=200) + timedelta(days=rnd.randint(0, 900))
        signups[cid] = signup
        name = f"{rnd.choice(FIRST_NAMES)} {rnd.choice(LAST_NAMES)}"
        premium = "yes" if rnd.random() < 0.25 else "no"
        customers.append([str(cid), name, typed_city, pincode, _dmy(signup), premium])
    tables["customers"] = customers

    products = [["Product ID", "Name", "Category", "Price (INR)"]]
    prices: dict[int, Decimal] = {}
    pid = 100
    for category, items in CATALOG.items():
        for name, price in items:
            pid += 1
            prices[pid] = Decimal(price)
            products.append([str(pid), name, category, _inr(Decimal(price))])
    tables["products"] = products

    orders = [["Order ID", "Customer ID", "Order Date", "Status", "Channel"]]
    items = [["Order ID", "Line No", "Product ID", "Quantity", "Unit Price", "Discount"]]
    span = (END - START).days
    order_ids = sorted(rnd.sample(range(10_001, 99_999), N_ORDERS))
    order_dates = sorted(START + timedelta(days=rnd.randint(0, span)) for _ in order_ids)
    for oid, odate in zip(order_ids, order_dates, strict=True):
        cid = rnd.choice([c for c, s in signups.items() if s <= odate] or [1])
        status = rnd.choices(STATUSES, weights=STATUS_WEIGHTS)[0]
        if (END - odate).days < 10:  # the most recent orders are not delivered yet
            status = rnd.choice(["placed", "shipped"])
        channel = rnd.choices(list(CHANNELS), weights=list(CHANNELS.values()))[0]
        orders.append([str(oid), str(cid), _dmy(odate), status, channel])

        picked = rnd.sample(
            sorted(prices), k=rnd.choices([1, 2, 3, 4], weights=[40, 30, 20, 10])[0]
        )
        for line_no, product in enumerate(picked, start=1):
            qty = rnd.choices([1, 2, 3, 5], weights=[70, 18, 8, 4])[0]
            unit = prices[product]
            discount = Decimal(0)
            if rnd.random() < 0.3:
                pct = Decimal(rnd.choice([5, 10, 15])) / 100
                discount = (unit * qty * pct).quantize(Decimal(1), ROUND_HALF_UP)
            items.append(
                [str(oid), str(line_no), str(product), str(qty), _inr(unit), _inr(discount)]
            )
    tables["orders"] = orders
    tables["order_items"] = items
    return tables


def _dmy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _inr(v: Decimal) -> str:
    return f"{v:,.2f}"  # 1,299.00


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    for name, rows in generate(args.seed).items():
        with (args.dest / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"  {name}.csv  {len(rows) - 1:>4} rows")
    print(f"✓ wrote {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
