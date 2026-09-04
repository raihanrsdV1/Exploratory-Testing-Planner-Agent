# Swag Labs (SauceDemo) — Requirements Specification

Version 1.0 · System under test: `https://www.saucedemo.com`

## Scope

Swag Labs is a demonstration e-commerce storefront published by Sauce Labs for
the express purpose of exercising automated testing tools. It holds no real
customer data, takes no real payment, and ships no real goods, so every user
journey may be driven to completion.

This document specifies the behaviour the application is **expected** to exhibit.
Where the implementation deviates, that deviation is a defect to be reported.

---

## 1. Authentication

### 1.1 Accepted credentials
DETAILED DESCRIPTION: The application shall present a login form accepting a
username and a password. The following usernames shall be accepted, all with the
password `secret_sauce`:

- `standard_user`
- `problem_user`
- `performance_glitch_user`
- `error_user`
- `visual_user`

Apart from response time (see 1.5), **every accepted account shall present the
same catalogue, the same controls, and the same behaviour throughout the
application.** A difference in rendering or behaviour between two accepted
accounts is a defect.

RATIONALE: Account identity governs who is shopping, not what the storefront does.

### 1.2 Rejected credentials
DETAILED DESCRIPTION: The application shall reject a login where:
- the username is empty — the error shall state that a username is required;
- the password is empty — the error shall state that a password is required;
- the username is not a known account, or the password is not `secret_sauce` —
  the error shall state that the username and password do not match.

Each rejection shall display an error message on the login page, and shall not
navigate away from it.

### 1.3 Locked-out account
DETAILED DESCRIPTION: The username `locked_out_user` shall be refused with an
error stating the user has been locked out, even when the password is correct.

### 1.4 Session and access control
DETAILED DESCRIPTION: Requesting an authenticated URL (for example
`/inventory.html`, `/cart.html`, `/checkout-step-one.html`) while signed out
shall refuse access and return the visitor to the login page with an error. It
shall not render the page's contents.

### 1.5 Sign-in performance
DETAILED DESCRIPTION: Sign-in shall complete within 2 seconds under normal
conditions. A materially slower sign-in is a performance defect and shall be
reported with the measured duration.

---

## 2. Product catalogue

### 2.1 Listing
DETAILED DESCRIPTION: After sign-in the application shall display the products
page, titled "Products", listing every product. Each entry shall show a name, a
description, a price, an image, and an "Add to cart" control.

### 2.2 Product images
DETAILED DESCRIPTION: Each product shall display **its own** image. Two different
products showing the same image is a defect. An image that fails to load is a
defect.

### 2.3 Prices
DETAILED DESCRIPTION: Every product shall display a price in the form `$N.NN`. A
missing, zero, negative, or non-numeric price is a defect. The price shown on the
listing, in the cart, and in the checkout summary shall be identical for the same
product.

### 2.4 Product detail
DETAILED DESCRIPTION: Activating a product name or image shall open that
product's detail page, showing the name, description, price and image of **the
product that was activated**. A detail page showing a different product than the
one selected is a defect. The page shall offer a control returning to the
listing.

### 2.5 Sorting
DETAILED DESCRIPTION: The listing shall provide a sort control with four options,
each of which shall reorder the listing accordingly:

- "Name (A to Z)" — ascending alphabetical by product name
- "Name (Z to A)" — descending alphabetical by product name
- "Price (low to high)" — ascending numeric by price
- "Price (high to low)" — descending numeric by price

The selected option shall remain selected after the listing re-renders. A sort
that leaves the order unchanged, or produces an order not matching the selected
option, is a defect.

---

## 3. Cart

### 3.1 Adding
DETAILED DESCRIPTION: Activating "Add to cart" for a product shall add exactly
that product to the cart, increment the cart badge by exactly one, and replace
that product's control with "Remove".

### 3.2 Removing
DETAILED DESCRIPTION: Activating "Remove" shall remove exactly that product,
decrement the cart badge by exactly one, and restore the "Add to cart" control.
Removing the last item shall leave the cart empty and the badge shall disappear
rather than display zero.

### 3.3 Cart contents
DETAILED DESCRIPTION: The cart page shall list every product added, each with its
name, description, price, and quantity, and no product that was not added. The
number of line items shall equal the cart badge count.

### 3.4 Persistence
DETAILED DESCRIPTION: Cart contents shall survive navigation between the listing,
a product detail page, and the cart, and shall survive a page reload.

### 3.5 Continue shopping
DETAILED DESCRIPTION: The cart page shall offer a control returning to the
product listing with the cart contents intact.

---

## 4. Checkout

### 4.1 Entry
DETAILED DESCRIPTION: The cart page shall offer a "Checkout" control leading to a
form requesting first name, last name, and postal code.

### 4.2 Field validation
DETAILED DESCRIPTION: The checkout form shall refuse to continue and shall
display an error naming the missing field when:
- first name is empty — "First Name is required";
- last name is empty — "Last Name is required";
- postal code is empty — "Postal Code is required".

Validation shall be evaluated on every submission, not only the first.

### 4.3 Order summary
DETAILED DESCRIPTION: On continuing, the application shall display an overview
listing every cart item with its price, an item total equal to the sum of the
item prices, a tax amount, and a total equal to item total plus tax. An
arithmetic inconsistency between these figures is a defect.

### 4.4 Completion
DETAILED DESCRIPTION: Activating "Finish" shall display an order-confirmation
page. After completion the cart shall be empty and the cart badge shall not
display a count.

### 4.5 Cancellation
DETAILED DESCRIPTION: Cancelling at any checkout step shall return the visitor to
the preceding page with the cart contents unchanged.

---

## 5. Navigation and presentation

### 5.1 Menu
DETAILED DESCRIPTION: Every authenticated page shall offer a menu containing "All
Items", "About", "Logout" and "Reset App State". "All Items" shall return to the
listing. The menu shall be dismissible without activating an entry.

### 5.2 Responsive layout
DETAILED DESCRIPTION: At a viewport width of 375 CSS pixels the listing and cart
shall remain usable: no horizontal page scrolling and no control rendered outside
the viewport.

### 5.3 Client health
DETAILED DESCRIPTION: No ordinary user journey — sign in, browse, sort, add to
cart, check out — shall produce an uncaught JavaScript exception or a request
returning HTTP 5xx.

---

## 6. Out of scope

The following shall not be exercised by an automated agent, because each one
destroys the state the remaining tests depend on:

- **Logout**, which ends the session for every subsequent test in the batch.
- **Reset App State**, which silently empties the cart mid-journey and would make
  an unrelated test appear to fail.

RATIONALE: Both are ordinary, non-destructive features of the application and are
perfectly valid to test by hand. They are excluded here only because a batch
shares one browser session, so one activation invalidates everything after it.
