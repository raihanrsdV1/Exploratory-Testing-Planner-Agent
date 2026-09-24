# BookStack (WebTestPilot benchmark) — Requirements Specification

Version 1.0 · System under test: `http://localhost:8081`

## Scope

BookStack is an open-source documentation platform, run here from the
WebTestPilot seeded-defect benchmark. One requirement is in scope for this
profile: creating a book. WebTestPilot pairs it with a known, injectable
defect (`benchmark/bookstack/bugs/create_book.js`) that corrupts the book's
description or removes it from the recents list on a second visit to the
Books page — the ground truth this campaign is run against.

---

## 1. Book creation

### 1.1 Create a book
DETAILED DESCRIPTION: From the dashboard, the user shall be able to open
Books, click "Create New Book", enter a name and description, and save it.
After saving, the book's page shall show the entered name as its heading.

RATIONALE: Creating a book is the platform's most basic authoring action;
every other content type (chapters, pages) lives under a book.

### 1.2 Book listing stays consistent
DETAILED DESCRIPTION: After creating a book, returning to the Books page
shall show that book under "Recently Viewed"/"New Books", **with the same
title and description the user entered.** The description shall not change,
and the entry shall not disappear, from being viewed again.

RATIONALE: A book's listing is the only record most users see of what they
just created — if the title, description, or presence of the entry changes
between visits with no user action to explain it, that is a defect.
