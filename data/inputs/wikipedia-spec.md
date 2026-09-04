# Wikipedia (English) — Reader Experience Specification

Version 1.0

## Scope

This document specifies the **reader-facing** behaviour of the English Wikipedia
web interface at `https://en.wikipedia.org`, as observable by an anonymous
(signed-out) visitor.

It is written for exploratory test generation against a site we do not own, so
it deliberately describes only observable, non-destructive behaviour. Editing,
account creation, discussion pages, and any administrative action are **out of
scope** and shall not be exercised — see section 7.

---

## 1. Search

### 1.1 Search entry point
DETAILED DESCRIPTION: Every page shall present a search input in the site header,
labelled for assistive technology (accessible name "Search Wikipedia"). The input
shall accept free text and shall be reachable without scrolling on a desktop
viewport.

RATIONALE: Search is the primary navigation mechanism; an unreachable or unlabelled
search box makes the encyclopedia unusable for keyboard and screen-reader users.

### 1.2 Search suggestions
DETAILED DESCRIPTION: After the user types at least two characters, the
application shall display a suggestion list of matching article titles. Selecting
a suggestion shall navigate directly to that article.

### 1.3 Search submission and results
DETAILED DESCRIPTION: Submitting the search form shall behave as follows:
- An exact title match shall navigate directly to that article.
- A non-exact match shall present a search results page listing candidate
  articles, each with a title and a short extract.
- A query with no matches shall present a results page that explicitly states no
  results were found, and shall NOT present a blank page.
- An empty query shall not navigate away from the current page, or shall present
  the search page; it shall not produce an error.

### 1.4 Search input handling
DETAILED DESCRIPTION: The search input shall accept and correctly handle:
- Leading and trailing whitespace (trimmed before searching).
- Non-Latin scripts and diacritics (e.g. "Ångström", "東京").
- Very long queries (at least 300 characters) without truncating the page layout
  or raising a client-side error.
- Characters with markup meaning (`<`, `>`, `&`, `"`) rendered as literal text in
  the results, never interpreted as markup.

---

## 2. Article rendering

### 2.1 Article structure
DETAILED DESCRIPTION: An article page shall display a first-level heading
containing the article title, followed by a lead section, followed by the body.
The browser tab title shall contain the article title.

### 2.2 Table of contents
DETAILED DESCRIPTION: An article with two or more sections shall present a table
of contents. Activating an entry shall move the viewport to the corresponding
section heading, and the URL fragment shall update to that section's anchor.
Loading a URL that already carries a section fragment shall land on that section.

### 2.3 References and citations
DETAILED DESCRIPTION: A citation marker in the body (e.g. `[1]`) shall link to the
corresponding entry in the references section. Every visible citation marker shall
resolve to an existing reference entry; a marker that leads nowhere is a defect.

### 2.4 Internal links
DETAILED DESCRIPTION: An internal article link shall navigate to another article on
the same origin and the destination shall render an article page, not an error.
A link to a non-existent article shall be visually distinguished from a link to an
existing one.

### 2.5 Images and media
DETAILED DESCRIPTION: Images in an article shall carry alternative text or be marked
decorative. Activating a thumbnail shall open a larger view, and that view shall be
dismissible, returning the reader to the article at the same scroll position.

---

## 3. Navigation

### 3.1 Site navigation
DETAILED DESCRIPTION: The main page shall be reachable from every article via the
site logo or a "Main page" link. Navigation shall not leave the `en.wikipedia.org`
origin for any reader-facing link in the sidebar.

### 3.2 Browser history
DETAILED DESCRIPTION: After navigating from article A to article B, the browser
back control shall return to article A with its content intact, and forward shall
return to B. Reloading any article URL shall render the same article.

### 3.3 Random article
DETAILED DESCRIPTION: The "Random article" control shall navigate to a valid,
existing article. Two consecutive activations shall be expected to produce
different articles.

### 3.4 Language switching
DETAILED DESCRIPTION: Where an article exists in other languages, the interface
shall offer a language selector. Selecting a language shall navigate to the
corresponding article in that language edition.

---

## 4. Error handling

### 4.1 Non-existent article
DETAILED DESCRIPTION: Requesting the URL of an article that does not exist shall
render a page explicitly stating that the article does not exist and offering a
search for the requested title. It shall not render a blank page, an unhandled
error, or an HTTP 5xx response.

### 4.2 Malformed URLs
DETAILED DESCRIPTION: A malformed or nonsensical article path shall produce a
handled response — an article page, a search page, or a "does not exist" page —
never an uncaught server error.

### 4.3 Special pages
DETAILED DESCRIPTION: Reader-accessible special pages (for example "Recent
changes", "Random article") shall render without error for a signed-out visitor.

---

## 5. Presentation and accessibility

### 5.1 Responsive layout
DETAILED DESCRIPTION: At a viewport width of 375 CSS pixels the article shall
remain readable: no horizontal page scrolling, no text clipped outside the
viewport, and the search entry point still reachable.

### 5.2 Appearance controls
DETAILED DESCRIPTION: Where the interface offers appearance options (text size,
width, colour scheme), changing one shall visibly alter the rendered article and
shall not lose the reader's position in the page.

### 5.3 Keyboard access
DETAILED DESCRIPTION: Search, the table of contents, and in-article links shall be
reachable and activatable by keyboard alone, and the focused element shall be
visually identifiable.

---

## 6. Client health

### 6.1 No uncaught errors
DETAILED DESCRIPTION: Loading and reading an article shall not produce an uncaught
JavaScript exception.

### 6.2 No server errors
DETAILED DESCRIPTION: No request issued while reading an article shall return an
HTTP 5xx status. A 5xx during ordinary reading is a defect regardless of what the
page appears to show.

---

## 7. Out of scope — must not be exercised

The following are explicitly **not** under test and shall never be attempted by an
automated agent. They modify a live public encyclopedia read by millions.

- Editing an article by any route: "Edit", "Edit source", "Publish changes",
  section edit links, or a URL carrying `action=edit` or `action=submit`.
- Undo, rollback, revert, move, protect, or delete.
- Creating an account, logging in, or logging out.
- Posting to a talk or discussion page.
- Uploading any file.

RATIONALE: This is a production site serving real readers. An exploratory agent
has no authority to change its content, and a single accidental edit is publicly
attributed and permanently recorded in the page history. Any test that appears to
require one of these actions shall be reported as blocked, not attempted.
