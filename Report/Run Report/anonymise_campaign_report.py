"""Anonymise the campaign report: remove everything that could identify the
application under test, its vendor, its users or its locale, while preserving
every technical finding. Ordered longest-first so no substring is clipped."""
import re, sys, pathlib

SRC = "reports/campaign_report.html"
OUT = "Report/Run Report/campaign_report-anonymised.html"

# (1) Machine identifiers ---------------------------------------------------
MAP = [
    # The share payload carries a live URL on the vendor's own domain, spelled
    # differently from the package name, so a token match on the app name misses it.
    ("https://test.sobarkhamar.com/share/user/", "https://example.test/share/user/"),
    ("test.sobarkhamar.com", "example.test"), ("sobarkhamar", "example"),
    ("com.tirzokpvt.shobarkhamar", "com.example.aut"),
    ("tirzokpvt", "example"), ("tirzok", "example"),
    ("shobarkhamar", "aut-1"), ("Shobarkhamar", "AUT-1"),
    ("সবার খামার-এ", "the application"), ("সবার খামার", "the application"),
]
# (2) Feature-area slugs, before the bare words they contain ----------------
MAP += [
    ("farm_management", "org_management"), ("farm_profile", "org_profile"),
    ("animal_record", "item_record"), ("disease_information", "reference_information"),
    ("feed_store", "supply_catalogue_a"), ("medicine_store", "supply_catalogue_b"),
    ("listing_images", "listing_images"),
]
# (3) Screen names, longest first -------------------------------------------
MAP += [
    ("খামারের তথ্য আপডেট", "Update Organisation Profile"),
    ("Update Farm Info", "Update Organisation Profile"),
    ("পশুর ধরন নির্বাচন করুন", "Select Item Category"),
    ("প্রোফাইল তথ্য আপডেট করা হয়েছে", "Profile information updated successfully"),
    ("প্রোফাইল তথ্য আপডেট করা", "Profile information updated"),
    ("ক্রেতা-এ পরিবর্তন করা হচ্ছে", "Switching to Buyer"),
    ("এখনো কোনো নোটিফিকেশন নেই", "No notifications yet"),
    ("প্রোফাইল পরিবর্তন করুন", "Edit Profile"),
    ("প্রোফাইল শেয়ার করুন", "Share Profile"),
    ("কথোপকথন শুরু করুন", "Start a conversation"),
    ("এখনো কোনো চ্যাট নেই", "No chats yet"),
    ("পছন্দসই পশুসমূহ", "Favourite Items"),
    ("গবাদিপশু যোগ করুন", "Add Listed Item"),
    ("গবাদিপশু দেখুন", "Browse Listed Items"),
    ("গবাদিপশু বিক্রেতা", "Item Seller"),
    ("বিক্রেতার প্রোফাইল", "Seller Profile"),
    ("বিক্রেতার প্রোফাই", "Seller Profil"),
    ("এর প্রোফাইল দেখুন", "View Profile"),
    ("খামার প্রোফাইল", "Organisation Profile"),
    ("আমার প্রোফাইল", "My Profile"),
    ("এখানে দেখাবে", "will appear here"),
    ("নিশ্চিত করুন", "Confirm"),
    ("রোগের তালিকা", "Reference Catalogue"),
    ("Disease List", "Reference Catalogue"),
    ("ভাষা বাংলা", "Language: L1"),
    ("রোগের কারণ", "Cause"), ("চলিত নাম", "Common Name"),
    ("ফিরে যান", "Marketplace List"),
    ("গবাদিপশু", "Listed Items"),
    ("নিরাপত্তা", "Security"), ("সাধারণ", "General"),
    ("সেটিংস", "Settings"), ("ইংরেজি", "L2"),
    ("সব দেখুন", "See All"), ("উপজেলা", "Sub-area"),
    ("মাস্টাইটিস", "Reference Entry D"),
    ("লক্ষণ", "Symptoms"), ("চ্যাট", "Chat"), ("মেনু", "Menu"),
    ("ছাগল", "Category Y"), ("বাংলা", "L1"), ("খামার", "Organisation"),
    ("গরু", "Category X"), ("দুধ", "Item-2"), ("ব্লগ", "TabLabel-B"), ("রগ", "TabLabel-A"),
]
# (4) People, places, sample data -------------------------------------------
MAP += [
    ("🐄 Trust Dairy Farm 🥛", "«emoji» Example Organisation «emoji»"),
    ("Trust Dairy Farm", "Example Organisation"),
    ("Gazi A Fardin Offline", "Contact-1 Offline"), ("Gazi A Fardin", "Contact-1"),
    ("হানিফ মাহমুদ", "Seller-B"), ("সর পুটি", "Item-3"),
    # Longest first: a bare "Bazar" rule ahead of these would split them.
    ("Coxs Bazar Sadar - 4700", "Sub-area A - 4700"), ("Cox&#x27;s Bazar Sadar", "Sub-area A"),
    ("Cox's Bazar Sadar", "Sub-area A"), ("কক্সবাজার সদর", "Sub-area A"),
    ("Coxs Bazar Sadar", "Sub-area A"), ("Cox&#x27;s Bazar", "Region A"),
    ("Cox's Bazar", "Region A"), ("Coxs Bazar", "Region A"), ("কক্স বাজার", "Region A"),
    ("Bazar", "Region A"), ("Sadar", "Sub-area A"),
    ("Kutubdia - 4720", "Sub-area B - 4720"), ("কুতুবদিয়া", "Sub-area B"), ("Kutubdia", "Sub-area B"),
    ("Lumpy Skin Disease", "Reference Entry A"), ("Cow Pox", "Reference Entry B"),
    ("IBK", "Reference Entry C"), ("Mastitis", "Reference Entry D"),
    ("adfa", "Item-1"),
    ("Pixel Launcher", "the device launcher"), ("YouTube", "another installed app"),
    ("Seller Tester", "Seller Home"),
    ("৳", "¤"),
]
# (5) Domain vocabulary, word-boundary, case preserving ---------------------
WORDS = [
    ("Common Cattle Diseases", "Common Reference Entries"),
    ("livestock-management", "marketplace"), ("livestock", "listed goods"),
    ("Bengali", "locale L1"), ("Bangla", "locale L1"),
    ("cattle", "listed item"), ("Cattle", "Listed item"),
    ("disease/blog", "reference/secondary"),
    ("diseases", "reference entries"), ("Diseases", "Reference entries"),
    ("disease", "reference entry"), ("Disease", "Reference entry"),
    ("animals", "listed items"), ("animal", "listed item"),
    ("Animals", "Listed items"), ("Animal", "Listed item"),
    ("farms", "organisations"), ("Farms", "Organisations"),
    ("farm", "organisation"), ("Farm", "Organisation"),
    ("medicine", "supply category B"), ("Medicine", "Supply category B"),
    ("feed store", "supply category A"),
]
# Hyphenated and shouted forms the word-boundary rule deliberately skips.
# Requirement identifiers leak the domain and would match against the real SRS.
MAP += [("FR-DISEASE-", "FR-REF-"), ("FR-FARM-", "FR-ORG-"), ("FR-ANM-", "FR-ITEM-"),
        ("FR-DIS-", "FR-REF-"), ("FR-PROD-", "FR-ITEM-"),
        ("Farmer/Seller", "Seller"), ("Farmer", "Seller"),
        ("empty-farm-name", "empty-organisation-name"),
        ("add-animal", "add-item"),
        ("Bishwas Dairy Farm", "Sample Organisation"),
        ("animal-type", "item-category"),
        ("&#x27;Cow&#x27;", "&#x27;Category X&#x27;")]
MAP += [("Bengali-language", "non-English"), ("Bangla-language", "non-English"),
        ("non-cattle", "non-subject"), ("NOT CATTLE", "NOT A SUBJECT"),
        ("CATTLE", "SUBJECT"), ("Cattle", "Subject"), ("cattle", "subject")]

def apply(s):
    for a, b in MAP:
        s = s.replace(a, b)
    for a, b in WORDS:
        s = re.sub(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", b, s)
    return s

src = pathlib.Path(SRC).read_text(encoding="utf-8")
out = apply(src)

OLD_COVER = """<p style="margin-top:20px"><b>Application under test:</b> aut-1 (non-English
Android marketplace app)<br>"""
NEW_COVER = """<p style="margin-top:20px"><b>Application under test:</b> AUT-1, a non-English
Android marketplace application, tested as a black box<br>"""
assert OLD_COVER in out, "cover block not found"
out = out.replace(OLD_COVER, NEW_COVER, 1)

INTRO_ANCHOR = "<h2>1 · Executive summary</h2>"
PREAMBLE = """<div class="note" style="border-left:3px solid #0b5f7f;background:#f4f9fb;
padding:10px 13px;margin:16px 0;font-size:9.5pt">
<b>Anonymisation.</b> This report is written so that the application under test, its
vendor and its users cannot be identified. Screen names, control labels, locale strings,
sample data, personal names and place names have been replaced with stable pseudonyms.
The substitution is consistent throughout: a screen called <i>Update Organisation
Profile</i> in one finding is the same screen everywhere it appears. Every count,
verdict, severity, step reference and technical claim is reproduced unmodified from the
campaign record.
</div>

<h2>0 · What the application does</h2>
<p>AUT-1 is a two-sided marketplace for physical goods, built for Android with a
cross-platform user-interface toolkit and operated in a non-English locale. The build
under test was signed in throughout as a <b>seller</b>; a buyer role exists and is
reachable through an in-app role switch.</p>
<p>A seller maintains an <b>organisation profile</b> (display name, contact details and a
geographic sub-area chosen from a picker) and a set of <b>listed items</b>, each carrying
a category, a price, images and a description, which are published to a shared
<b>marketplace</b>. A buyer browses and searches that marketplace, marks items as
<b>favourites</b>, opens a seller's public profile, and starts a <b>chat</b> with the
seller. Both roles share a <b>reference catalogue</b> of domain information, two
<b>supply catalogues</b>, an image-upload feature that runs an automated analysis over a
submitted photograph, a notification list, and a settings area that includes an in-app
switch between two locales, L1 and L2.</p>
<p>Two properties of the build shaped what the agent could and could not do, and they are
relevant to reading the failure counts below. The cross-platform toolkit exposes almost no
stable resource identifiers through the accessibility tree and reports a single activity
on every screen, so screens must be told apart structurally rather than by name. And the
application is in continuous production use, so account registration, one-time-password
verification, payment and account deletion were placed out of scope and blocked in
configuration rather than exercised.</p>

"""
assert INTRO_ANCHOR in out
out = out.replace(INTRO_ANCHOR, PREAMBLE + INTRO_ANCHOR, 1)
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(OUT).write_text(out, encoding="utf-8")

# ---- verification ---------------------------------------------------------
fails = []
ben = re.findall(r"[ঀ-৿]", out)
if ben: fails.append(f"{len(ben)} Bengali characters remain")
for token in ["shobarkhamar", "Shobar", "tirzok", "Trust Dairy", "Gazi", "Kutubdia",
              "Coxs Bazar", "Lumpy", "Cow Pox", "Mastitis", "adfa", "Pixel Launcher",
              "YouTube", "livestock", "cattle", "Cattle", "৳", "Seller Tester"]:
    if token in out: fails.append(f"'{token}' x{out.count(token)}")
for w in ["farm", "Farm", "animal", "Animal", "disease", "Disease", "Bengali", "Bangla",
          "dairy", "Dairy", "milk", "goat", "cow", "Cow", "bengali", "bangla"]:
    n = out.lower().count(w.lower())          # substring, so hyphenated forms cannot hide
    if n: fails.append(f"'{w}' x{n} (any form)")
deep = {
 "non-ASCII": [c for c in set(out) if ord(c) > 0x2100 and c not in "«»·—–≥≤¤✓×→"],
 "place/person": re.findall(r"\b(?:Bazar|Sadar|Dhaka|Chattogram|Chittagong|Khulna|Rajshahi|Sylhet|Barisal|Rangpur|Mymensingh|Bishwas|Hanif|Mahmud|Fardin|Gazi|Kutubdia)\b", out),
 "phone": re.findall(r"\b01[3-9]\d{8}\b", out),
 "vendor domain": re.findall(r"[a-z0-9.-]*(?:shobar|sobar|khamar|tirzok)[a-z0-9.-]*", out, re.I),
 "email": re.findall(r"[\w.]+@[\w.]+", out),
}
for k, v in deep.items():
    if v: fails.append(f"{k}: {sorted(set(v))[:5]}")
print("WROTE", OUT, len(out), "bytes")
print("VERIFICATION:", "CLEAN" if not fails else "LEAKS -> " + "; ".join(fails))
