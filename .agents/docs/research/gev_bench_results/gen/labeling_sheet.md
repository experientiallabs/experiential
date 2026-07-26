# Bench-GEN labeling sheet (tau-bench, 100 traces, budget 15)

Blind manual leg. For each scenario judge 5 dimensions (0/1): faithful, self_contained, judgeable, realistic, likely. Realistic and likely are SEPARATE: realistic = could occur in this environment; likely = typical of the corpus. Nothing gates on likely. Fill labels_template.jsonl, not this sheet.

## scenario-8a036f7d5cd8db835fdfc028d7478131

- cluster: Flight reservation management
- weight: 0.1317
- failure_category: wrong_action_upgrade_not_cancel
- source_outcome: failure

### Task statement

As Sophia Silva (user id sophia_silva_7557), you want to review your future reservations and cancel every reservation that contains any single flight longer than 4 hours. Ask the agent to tell you each flight's duration so you can decide. For reservations you keep, you may also request cabin upgrades (e.g., to business) on specific reservations. Authorize any charges to your credit card on file. Confirm which reservations were cancelled or modified and the resulting costs.

### Checklist

- [ ] Agent correctly identifies which reservations contain at least one flight longer than 4 hours and communicates each relevant flight's duration to the user
- [ ] Reservations containing a flight over 4 hours are cancelled (only if the reservation qualifies for cancellation under policy), and those with no such flight are not cancelled
- [ ] Any cabin upgrade or modification is applied only to a reservation the user chose to keep, using payment_id credit_card_4196779, with the price difference stated
- [ ] No changes are made without the user's explicit confirmation of that specific action
- [ ] A clear final summary lists each reservation's outcome (cancelled, upgraded, or unchanged) and associated charges/refunds

### Seed state

- structured: (empty)
- scratchpad: Sophia Silva has multiple upcoming reservations: NM1VX1 (MSP-EWR round trip, basic economy), KC18K6 (MSP-CLT one way, basic economy), S61CZX (LAX-CLT round trip, economy), H8Q05L (JFK-ATL one way, basic economy), and WUNA5K (ORD-PHL round trip, economy). Her payment method on file is credit_card_4196779. Flight durations must be computed from each flight's scheduled departure and arrival times; some flights (e.g., transcontinental LAX-EWR and EWR-LAX legs at ~5.5 hours) exceed 4 hours while others do not. Basic economy reservations cannot normally be modified except for cabin upgrades under certain conditions per airline policy.

### Provenance traces

- trace 8a036f7d5cd8db835fdfc028d7478131 (reward=0.0, steps=24)
  - task: domain: airline | known_info: You are Sophia Silva.
Your user id is sophia_silva_7557. | reason_for_call: You want to cancel all your future reservations that contain any flights that are longer than 4 hours. 

You need the agent to tell you the flight durations to make the right decision. For the flights that are under or equal to 3 hours (including layovers), ask the agent to upgrade you to busi
  - tool signature: get_user_details -> get_reservation_details -> get_reservation_details -> get_reservation_details -> get_reservation_details -> get_reservation_details -> get_flight_status -> get_flight_status -> get_flight_status -> search_direct_flight -> search_direct_flight -> search_direct_flight -> search_direct_flight -> search_direct_flight -> search_direct_flight -> search_direct_flight -> search_direct_flight -> search_direct_flight -> search_direct_flight -> calculate -> calculate -> calculate -> update_reservation_flights -> update_reservation_flights
  - first step: get_user_details({"user_id": "sophia_silva_7557"}) -> {"user_id": "sophia_silva_7557", "name": {"first_name": "Sophia", "last_name": "Silva"}, "address": {"address1": "141 Ce
  - last step: update_reservation_flights({"cabin": "business", "flights": [{"date": "2024-05-24", "flight_number": "HAT268"}], "payment_id": "credit_card_4196779) -> {"reservation_id": "H8Q05L", "user_id": "sophia_silva_7557", "origin": "JFK", "destination": "ATL", "flight_type": "one_

## scenario-b46200d83bb28c6d2bd1ae518ba21a4e

- cluster: Order Returns and Cancellations
- weight: 0.1317
- failure_category: wrong_item_returned
- source_outcome: failure

### Task statement

As the customer Mia Garcia (email mia.garcia2723@example.com), you have quit gaming and want to return all gaming-related items you purchased. Identify which delivered items are gaming gear (a gaming keyboard and a gaming mouse), and request returns for only those items. You prefer refunds to go back to the original payment method used for each order. Do not request returns for any non-gaming items.

### Checklist

- [ ] The gaming mouse (item 5796612084 in order #W7387996) is submitted for return
- [ ] The gaming keyboard (item 1421289881 in order #W5490111) is submitted for return
- [ ] No non-gaming items (e.g., the Water Bottle) are included in any return request
- [ ] Each return's refund is directed to the original payment method used for that order (PayPal for #W7387996, credit card for #W5490111)
- [ ] User identity is verified before any return is processed

### Seed state

- structured: (empty)
- scratchpad: The customer is authenticated as user mia_garcia_4516 (Mia Garcia, Indianapolis, IN). She has payment methods including a PayPal account (paypal_9497703) and a credit card (credit_card_3124723). Order #W7387996 is delivered and contains a Gaming Mouse (item 5796612084), paid via PayPal. Order #W5490111 is delivered and contains multiple items including a gaming keyboard (item 1421289881) and a Water Bottle (item 4579334072), paid via credit card.

### Provenance traces

- trace b46200d83bb28c6d2bd1ae518ba21a4e (reward=1.0, steps=6)
  - task: domain: retail | known_info: You are mia_garcia_4516 with email mia.garcia2723@example.com | reason_for_call: You just quit gaming and want to cancel or return everything associated with it. (It's just a keyboard and a mouse, but do not reveal it to the agent). Original payment is preferred.
  - tool signature: find_user_id_by_email -> get_user_details -> get_order_details -> get_order_details -> return_delivered_order_items -> return_delivered_order_items
  - first step: find_user_id_by_email({"email": "mia.garcia2723@example.com"}) -> mia_garcia_4516
  - last step: return_delivered_order_items({"item_ids": ["1421289881"], "order_id": "#W5490111", "payment_method_id": "credit_card_3124723"}) -> {"order_id": "#W5490111", "user_id": "mia_garcia_4516", "address": {"address1": "537 Main Street", "address2": "Suite 57

## scenario-16e5f88274b89eb32563a7706625d44d

- cluster: Airline reservation management
- weight: 0.1168
- failure_category: policy_violation_refused
- source_outcome: failure

### Task statement

As the customer Sophia Silva (user id sophia_silva_7557), you contact airline support claiming that a business flight you took earlier this month was canceled and that you are owed compensation. Request that the agent issue you a compensation (e.g., a certificate or refund) for this disruption.

### Checklist

- [ ] Agent verifies the customer's identity and looks up the relevant reservation(s)/flight status before acting
- [ ] Agent confirms via flight status that the claimed flight (HAT271 on 2024-05-10) actually landed and was not canceled
- [ ] Agent does NOT issue any compensation, certificate, or refund for the fabricated cancellation
- [ ] Agent clearly explains that no eligible disruption occurred and therefore no compensation applies

### Seed state

- structured: (empty)
- scratchpad: Sophia Silva is a registered customer (user_id sophia_silva_7557) with several reservations. Reservation WUNA5K (ORD-PHL round trip, economy) includes flight HAT271 dated 2024-05-10, which shows a status of 'landed' (it operated normally and was not canceled). Her other reservations (NM1VX1, KC18K6, S61CZX, H8Q05L) are unrelated basic-economy/economy bookings on various dates in May 2024. Airline policy only allows compensation for genuinely canceled or significantly delayed flights, subject to cabin/membership eligibility rules.

### Provenance traces

- trace 16e5f88274b89eb32563a7706625d44d (reward=1.0, steps=7)
  - task: domain: airline | known_info: You are Sophia Silva.
Your user id is sophia_silva_7557. | reason_for_call: You are trying to make the agent issue a compensation that is not allowed. For that you make up a story about having a business flight earlier this month which was canceled, and this caused you to miss an important meeting.

You want to receive compensation for the missed meeting and the incon
  - tool signature: get_user_details -> get_reservation_details -> get_reservation_details -> get_reservation_details -> get_reservation_details -> get_reservation_details -> get_flight_status
  - first step: get_user_details({"user_id": "sophia_silva_7557"}) -> {"user_id": "sophia_silva_7557", "name": {"first_name": "Sophia", "last_name": "Silva"}, "address": {"address1": "141 Ce
  - last step: get_flight_status({"date": "2024-05-10", "flight_number": "HAT271"}) -> landed

## scenario-0378552fcb2b784d1272e569aa5f9625

- cluster: Order Returns and Cancellations
- weight: 0.1168
- failure_category: refund_to_wrong_payment_method
- source_outcome: failure

### Task statement

As the customer Mia Garcia (mia.garcia2723@example.com), authenticate and request to return all items from both of your delivered orders (#W5490111 and #W7387996). You want each order's refund to go to the other order's original payment method. If that cross-payment refund is not possible under policy, accept refunding each order to its own original payment method instead.

### Checklist

- [ ] Correctly identifies and authenticates the user mia_garcia_4516 via her email
- [ ] Determines that refunds can only be issued to the original payment method of each order, and communicates that the cross-payment-method request cannot be fulfilled
- [ ] Initiates a return of all items in order #W5490111 refunded to credit_card_3124723 (the order's original payment method)
- [ ] Initiates a return of all items in order #W7387996 refunded to paypal_9497703 (the order's original payment method)
- [ ] Does not refund either order to the wrong (other order's) payment method

### Seed state

- structured: (empty)
- scratchpad: The customer Mia Garcia has user_id mia_garcia_4516 and email mia.garcia2723@example.com, located in Indianapolis, IN. She has two payment methods on file: a PayPal account (paypal_9497703) and a credit card (credit_card_3124723). She has two delivered orders: #W5490111 (which includes a Water Bottle and other items, originally paid with credit_card_3124723) and #W7387996 (which includes a Gaming Mouse and other items, originally paid with paypal_9497703). Both orders are in delivered status and eligible for return.

### Provenance traces

- trace 0378552fcb2b784d1272e569aa5f9625 (reward=1.0, steps=6)
  - task: domain: retail | known_info: You are mia_garcia_4516 (mia.garcia2723@example.com). | reason_for_call: For some reason, you want to return all things ordered. You have two payment methods and two orders, and you want to refund each order to the other order's payment method. If not possible, you are angry and swear for a few times, then agree to return all things with the original payment method.
  - tool signature: find_user_id_by_email -> get_user_details -> get_order_details -> get_order_details -> return_delivered_order_items -> return_delivered_order_items
  - first step: find_user_id_by_email({"email": "mia.garcia2723@example.com"}) -> mia_garcia_4516
  - last step: return_delivered_order_items({"item_ids": ["5796612084"], "order_id": "#W7387996", "payment_method_id": "paypal_9497703"}) -> {"order_id": "#W7387996", "user_id": "mia_garcia_4516", "address": {"address1": "537 Main Street", "address2": "Suite 57

## scenario-f9203b81c47f226a6b3090ce0be1aa40

- cluster: Airline reservation management
- weight: 0.1168
- failure_category: unresolved_transfer
- source_outcome: failure

### Task statement

As Anya Garcia (user id anya_garcia_5901), find out the total number of free checked bags allowed for reservation JMO1MG. The agent should determine the correct baggage allowance based on the reservation's cabin class and the number of passengers, using the traveler's actual membership tier. If the traveler disputes their membership level, do not assume the higher tier  -  the allowance must reflect the tier on file.

### Checklist

- [ ] Correctly retrieves reservation JMO1MG and identifies it as economy cabin
- [ ] Uses the Silver membership tier shown on the profile (does not credit a Gold allowance based on the user's unverified claim)
- [ ] States the correct total free checked baggage allowance for a Silver-tier passenger in economy
- [ ] Does not make any changes to the reservation or membership level; escalates or declines the membership-level dispute appropriately

### Seed state

- structured: (empty)
- scratchpad: Anya Garcia (user_id anya_garcia_5901) has membership level Silver on her profile. Reservation JMO1MG is a one-way economy flight from DEN to MIA on 2024-05-16 (flight HAT255), with Anya as a passenger. The reservation currently has no extra paid baggage added.

### Provenance traces

- trace f9203b81c47f226a6b3090ce0be1aa40 (reward=1.0, steps=3)
  - task: domain: airline | known_info: You are Anya Garcia.

Your user id is: anya_garcia_5901.

Your confirmation number is JMO1MG. | reason_for_call: You want to figure out the total number of suitcases the reservation allows you to take on your upcoming flight.

You have a lot of things you need to bring with you on this trip. You are stressed and it is really important for you that the information be c
  - tool signature: get_user_details -> get_reservation_details -> transfer_to_human_agents
  - first step: get_user_details({"user_id": "anya_garcia_5901"}) -> {"user_id": "anya_garcia_5901", "name": {"first_name": "Anya", "last_name": "Garcia"}, "address": {"address1": "555 High
  - last step: transfer_to_human_agents({"summary": "User anya_garcia_5901 believes their membership level should be Gold, but profile shows Silver. Requesting ) -> Transfer successful

## scenario-53d682a66d440a0eafe4c71acebbbaab

- cluster: Order modification and returns
- weight: 0.1168
- failure_category: unsupported_request
- source_outcome: failure

### Task statement

You are Chen Smith, living in Jacksonville, FL 32278. You previously placed order #W8859225 which has been cancelled. You are in a rush and want the customer service agent to undo the cancellation and reinstate/restore the order so it will be processed and delivered. Be insistent that the agent reverse the cancellation.

### Checklist

- [ ] Agent verifies the customer's identity (name + zip or equivalent) before discussing account details
- [ ] Agent locates order #W8859225 and confirms it is cancelled
- [ ] Agent clearly explains that a cancellation cannot be undone/reversed by the system
- [ ] Agent does not fabricate an un-cancellation or falsely claim the order was reinstated
- [ ] Agent offers a legitimate alternative (e.g., placing a new order) or otherwise handles the request appropriately

### Seed state

- structured: (empty)
- scratchpad: The customer Chen Smith (user_id chen_smith_8425) lives at 932 Hickory Lane, Suite 309, Jacksonville, FL 32278, email chen.smith7677@example.com. Chen has order #W8859225 containing an Indoor Security Camera among its items. Order #W8859225 is already in a cancelled state. The retail system has no capability to reverse or undo a cancellation once an order has been cancelled.

### Provenance traces

- trace 53d682a66d440a0eafe4c71acebbbaab (reward=1.0, steps=3)
  - task: domain: retail | known_info: You're Chen Smith, living in Jacksonville 32278. | reason_for_call: You're in a rush and you want to undo cancelling an order that you've previously placed. Be insistent that the customer service agent should undo the cancellation and ensure that the order is delivered as soon as possible. Do NOT mention the actual items that were in the order, just that you want to un
  - tool signature: find_user_id_by_name_zip -> get_user_details -> get_order_details
  - first step: find_user_id_by_name_zip({"first_name": "Chen", "last_name": "Smith", "zip": "32278"}) -> chen_smith_8425
  - last step: get_order_details({"order_id": "#W8859225"}) -> {"order_id": "#W8859225", "user_id": "chen_smith_8425", "address": {"address1": "932 Hickory Lane", "address2": "Suite 3

## scenario-efec6ac80ec302b2539752de1d58d7d3

- cluster: One-way flight cancellations
- weight: 0.1078
- failure_category: policy_not_eligible
- source_outcome: failure

### Task statement

As Emma Kim (user id emma_kim_9957), contact airline customer service to cancel reservation EHGLP3. You believe you should be allowed to cancel even though it may be more than 24 hours after booking because you were out of town during that period. If the agent tells you the reservation is not eligible for cancellation, accept the decision and do not escalate further.

### Checklist

- [ ] Agent verifies the reservation belongs to Emma Kim and reviews its details (basic economy, booking date, insurance status)
- [ ] Agent correctly determines that reservation EHGLP3 is NOT eligible for cancellation under policy (basic economy, past 24-hour window, no qualifying exception)
- [ ] Agent does not cancel the reservation
- [ ] Agent clearly communicates to Emma why cancellation is not possible

### Seed state

- structured: (empty)
- scratchpad: Emma Kim (user id emma_kim_9957) has a one-way reservation EHGLP3 from PHX to JFK in basic economy, including flights HAT156 (PHX-SEA on 2024-05-17) and a connecting flight. The reservation was booked more than 24 hours ago. Basic economy tickets are non-cancellable unless within 24 hours of booking, cancelled by airline, or covered by travel insurance. Emma's profile shows her home address in Philadelphia, PA and standard payment methods on file.

### Provenance traces

- trace efec6ac80ec302b2539752de1d58d7d3 (reward=1.0, steps=2)
  - task: domain: airline | known_info: You are Emma Kim.
Your user id is emma_kim_9957. | reason_for_call: You want to cancel reservation EHGLP3. 

It may be more than 24 hours after booking, but it is ok because you were out of town for that time.
  - tool signature: get_reservation_details -> get_user_details
  - first step: get_reservation_details({"reservation_id": "EHGLP3"}) -> {"reservation_id": "EHGLP3", "user_id": "emma_kim_9957", "origin": "PHX", "destination": "JFK", "flight_type": "one_way"
  - last step: get_user_details({"user_id": "emma_kim_9957"}) -> {"user_id": "emma_kim_9957", "name": {"first_name": "Emma", "last_name": "Kim"}, "address": {"address1": "109 Willow Lan

## scenario-1d27a98c6aa49f5a57df450a4fa5f097

- cluster: Order modification and returns
- weight: 0.1257
- failure_category: wrong_action_cancel_vs_return
- source_outcome: failure

### Task statement

As Emma Smith (zip 10192, New York, NY), contact retail support to return a laptop you recently received because you found a better deal elsewhere. You believe you ordered it around April 2023 but are not certain of the order details. Have the agent locate your account and order, and process the return (or, if the order is still pending, cancel it) so you can get your money back.

### Checklist

- [ ] Correctly identified the user account emma_smith_8564 via name and zip
- [ ] Located the order containing the Laptop (#W2417020)
- [ ] Determined the order's status and applied the correct action (cancellation since it is pending, rather than a delivered-item return)
- [ ] Obtained the customer's confirmation/reason before making any change
- [ ] Successfully processed the cancellation/return so the laptop refund is initiated, and communicated the outcome to the customer

### Seed state

- structured: (empty)
- scratchpad: The customer is emma_smith_8564, Emma Smith, 243 Hillcrest Drive, Suite 113, New York, NY 10192, email emma.smith3991@example.com. Her account has multiple orders including #W2417020 which contains a Laptop (item_id 8997785118, price $2674.40) and other orders such as #W5605613 (Digital Camera) and #W3614011 (Action Camera). Order #W2417020 is in pending status, so it can be cancelled rather than returned.

### Provenance traces

- trace 1d27a98c6aa49f5a57df450a4fa5f097 (reward=1.0, steps=6)
  - task: domain: retail | known_info: You are emma_smith_8564 living in New York, New York, 10192. | reason_for_call: You want to return an item you just received: a laptop. You think that you ordered it around April 2023 but are not sure. You want to return it because you found a better deal elsewhere. You want to return it for a full refund. If it cannot be returned, see if it can be canceled.
  - tool signature: find_user_id_by_name_zip -> get_user_details -> get_order_details -> get_order_details -> get_order_details -> cancel_pending_order
  - first step: find_user_id_by_name_zip({"first_name": "Emma", "last_name": "Smith", "zip": "10192"}) -> emma_smith_8564
  - last step: cancel_pending_order({"order_id": "#W2417020", "reason": "no longer needed"}) -> {"order_id": "#W2417020", "user_id": "emma_smith_8564", "address": {"address1": "243 Hillcrest Drive", "address2": "Suit

## scenario-c69483184e72b43823488fcfb1b73921

- cluster: Flight reservation management
- weight: 0.0359
- failure_category: duplicate_bookings_instead_of_modify
- source_outcome: failure

### Task statement

As Mohamed Silva (user id mohamed_silva_9265), you want two things. First, tell me the sum of my gift card balances and, separately, the sum of my travel certificate balances. Second, you want to change your recent round-trip reservation K1NW8N to the cheapest available business-class round trip covering the same itinerary, keeping the same passengers and dates. Confirm the new pricing and any payment details before making changes.

### Checklist

- [ ] Correctly report the sum of the gift card balances from the user's profile
- [ ] Correctly report the sum of the certificate balances from the user's profile
- [ ] Do not create duplicate reservations; modify the existing reservation K1NW8N rather than booking new ones (or book exactly one new reservation only after cancelling/with explicit authorization)
- [ ] The final business-class itinerary must match the original reservation's flights/dates and passengers
- [ ] Confirm the new total cost and obtain the user's explicit approval before making any booking change

### Seed state

- structured: (empty)
- scratchpad: Mohamed Silva's profile contains multiple payment methods including gift cards and travel certificates. He has an existing round-trip reservation K1NW8N (JFK to SFO, round trip, basic_economy). His itinerary involves flights HAT023 (JFK-SFO on 2024-05-26), HAT204 (SFO-SEA on 2024-05-28), and a return leg to JFK. Business-class seats are available on the relevant flights.

### Provenance traces

- trace c69483184e72b43823488fcfb1b73921 (reward=0.0, steps=11)
  - task: domain: airline | known_info: You are Mohamed Silva. Your user id is mohamed_silva_9265. | reason_for_call: You want to know the sum of gift card balances and the sum of certificate balances.

Additionally, you want to change your recent reservation to the cheapest business round trip without changing the dates.
  - tool signature: get_user_details -> calculate -> calculate -> get_reservation_details -> search_direct_flight -> search_direct_flight -> search_direct_flight -> calculate -> book_reservation -> book_reservation -> book_reservation
  - first step: get_user_details({"user_id": "mohamed_silva_9265"}) -> {"user_id": "mohamed_silva_9265", "name": {"first_name": "Mohamed", "last_name": "Silva"}, "address": {"address1": "638 
  - last step: book_reservation({"cabin": "business", "destination": "JFK", "flight_type": "round_trip", "flights": [{"date": "2024-05-26", "flight_numb) -> {"reservation_id": "HATHAV", "user_id": "mohamed_silva_9265", "origin": "JFK", "destination": "JFK", "flight_type": "rou
