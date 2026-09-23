# BookStack — Software Requirements Specification (for exploratory testing)

BookStack v25.02.1, deployed locally at http://localhost:8081 as part of the
WebTestPilot benchmark. This specification describes the behaviour a correct
BookStack deployment must exhibit. It is written from the application's
documented feature set and observable behaviour, for use as testing knowledge.

The tester is signed in as an administrator on a disposable local instance.

## 1. Content hierarchy

### FR-HIER-01 Shelves, books, chapters, pages [AUTH]
DETAILED DESCRIPTION: Content is organised as shelves containing books, books
containing chapters and pages, and chapters containing pages. Each level shall be
reachable by navigation from the one above, and every item shall display its own
name and its position in the hierarchy.

### FR-HIER-02 Creation [AUTH]
DETAILED DESCRIPTION: A signed-in user shall be able to create a shelf, book,
chapter and page. On save the new item's own view shall open, showing exactly the
name and description that were entered, unchanged.

### FR-HIER-03 Editing [AUTH]
DETAILED DESCRIPTION: Editing an item and saving shall persist every changed
field. Re-opening the item, and any listing that shows it, shall display the
updated values and never the previous ones.

### FR-HIER-04 Deletion [AUTH]
DETAILED DESCRIPTION: Deleting an item shall ask for confirmation, then remove it
from every listing that showed it. A deleted item shall not continue to appear in
listings, counts, or the items of its parent.

RATIONALE: A listing that still shows deleted content misleads the reader about
what exists, which is the failure this application most needs to avoid.

## 2. Listings and counters

### FR-LIST-01 Consistency with actual content [AUTH]
DETAILED DESCRIPTION: Every listing (Books, Shelves, a book's contents, search
results) shall agree with the content that actually exists. The name and
description shown for an item in a listing shall match that item's own page.

### FR-LIST-02 Recently created [AUTH]
DETAILED DESCRIPTION: "Recently Created" sections shall list the items most
recently created, newest first, and a newly created item shall appear there
immediately. Any count shown alongside shall equal the number of items listed.

### FR-LIST-03 Recently viewed [AUTH]
DETAILED DESCRIPTION: "Recently Viewed" shall list items the current user has
opened, most recent first. Opening an item shall place it at the head of that
list; an item never opened shall not appear.

### FR-LIST-04 Recent activity [AUTH]
DETAILED DESCRIPTION: The activity feed shall record each create, update and
delete with the acting user, the affected item, and a time. An action that
succeeded shall produce exactly one matching entry naming the correct item.

## 3. Favourites

### FR-FAV-01 Marking a favourite [AUTH]
DETAILED DESCRIPTION: A shelf, book, chapter or page shall be markable as a
favourite. The control shall then show the item as favourited, and the item shall
appear in the user's favourites listing.

### FR-FAV-02 Removing a favourite [AUTH]
DETAILED DESCRIPTION: Un-favouriting shall reverse FR-FAV-01 exactly: the control
returns to its unmarked state and the item leaves the favourites listing. The
state shall survive a page reload.

## 4. Search

### FR-SRCH-01 Finding content [AUTH]
DETAILED DESCRIPTION: Search shall return items whose name or content matches the
query, each result linking to the item it names. A query matching nothing shall
state that no matches were found rather than showing an empty or broken page.

### FR-SRCH-02 Result fidelity [AUTH]
DETAILED DESCRIPTION: A result shall describe the item it links to. Following a
result shall open exactly that item.

## 5. Pages and comments

### FR-PAGE-01 Page content [AUTH]
DETAILED DESCRIPTION: Page content entered in the editor shall be saved and
redisplayed unchanged, including formatting applied with the editor's controls.

### FR-PAGE-02 Comments [AUTH]
DETAILED DESCRIPTION: A comment added to a page shall appear on that page
attributed to its author, and shall persist across a reload.

### FR-PAGE-03 Templates [AUTH]
DETAILED DESCRIPTION: A page saved as a template shall be offered when creating a
new page, and applying it shall reproduce its content.

## 6. Organisation and settings

### FR-SORT-01 Sort rules [AUTH]
DETAILED DESCRIPTION: A sort rule shall be creatable and applicable to a book, and
applying it shall order that book's contents as the rule specifies.

### FR-SET-01 Settings persistence [AUTH]
DETAILED DESCRIPTION: A changed setting shall be saved, confirmed to the user, and
still in force after navigating away and returning.

## 7. General UI obligations

### FR-UI-01 Feedback on every action [AUTH]
DETAILED DESCRIPTION: An action that succeeds shall confirm visibly; one that
fails or is refused shall say so and why. No action shall complete silently in a
way that leaves the user unable to tell what happened.

### FR-UI-02 Validation [AUTH]
DETAILED DESCRIPTION: A required field left empty, or given only whitespace, shall
be refused with a message naming the field. The application shall not create an
item with an empty or whitespace-only name.

### FR-UI-03 No stale or fabricated data [AUTH]
DETAILED DESCRIPTION: Text shown for an item shall be the text stored for that
item. The interface shall never display a value that was not entered, nor omit an
item that exists, in any listing, count or summary.
