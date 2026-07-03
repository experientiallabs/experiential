# Merchant Guide to Optimizing Payment Processing and Minimizing Fees

Version 2.1 | Last Updated: November 1, 2024

## Table of Contents
1. Introduction
2. Account Type
3. Merchant Category Code
4. Authorization Characteristics Indicator
5. Understanding Payment Processing Fees
6. PIN Entry Attempt Limits
7. Reducing Fraud-Related Fees
8. Leveraging Data and Reporting
9. Appendix
   - Glossary
10. Contact Information

## 1. Introduction

As a valued merchant partner, our goal is to help you process transactions efficiently and cost-effectively while minimizing the risks associated with payment fraud. This guide provides best practices for configuring transactions, understanding pricing models, and reducing the potential for fraud-related fees.


## 2. Account Type

We categorize merchants into different account types based on their business model and industry classification. The following table outlines the various account types:

| Account Type | Description             |
|--------------|-------------------------|
| R            | Enterprise - Retail     |
| D            | Enterprise - Digital    |
| H            | Enterprise - Hospitality|
| F            | Platform - Franchise    |
| S            | Platform - SaaS         |
| O            | Other                   |

This categorization is used to provide more targeted support and services to merchants, and to facilitate more effective communication and collaboration between merchants and our team.

## 3. Merchant Category Code

The Merchant Category Code (MCC) is a four-digit code assigned to a merchant by the card networks, also known as schemes (e.g. Visa, Mastercard), to categorize their business type. The MCC is used to determine the type of business or industry a merchant is in, and is often used for risk assessment, fraud detection, and accounting purposes.

The MCC is typically assigned by the merchant's bank or payment processor, and is used to classify merchants into one of over 400 categories. Each category corresponds to a specific industry or business type, such as retail, restaurant, hotel, or healthcare.

The MCC is usually represented by a four-digit code, such as 5451 (Automated Fuel Dispensers) or 5812 (Automotive Parts and Accessories Stores). The first two digits of the MCC indicate the category, while the last two digits indicate the subcategory.

Here is an example of how the MCC might be used in a merchant's account information:

Merchant Name: ABC Car Dealership
Merchant Category Code (MCC): 5521 (Motor Vehicle Dealers - New and Used Cars)
Business Type: Retail
The MCC is an important piece of information for merchants, as it can affect their payment processing rates, fees, and other business operations.

You can find a complete list of MCC in the annexed file `merchant_category_codes.csv`. 

## 4. Authorization Characteristics Indicator (ACI)

The Authorization Characteristics Indicator is a field that facilitates the identification of the transaction flow submitted to the acquirer. This indicator provides a standardized method for describing the manner in which the transaction was sent to the acquirer.

The following table outlines the possible values for the Authorization Characteristics Indicator:

| Authorization Characteristic Indicator | Details                            |
|----------------------------------------|------------------------------------|
| A                                      | Card present - Non-authenticated   |
| B                                      | Card Present - Authenticated       |
| C                                      | Tokenized card with mobile device  |
| D                                      | Card Not Present - Card On File    |
| E                                      | Card Not Present - Recurring Bill Payment |
| F                                      | Card Not Present - 3-D Secure      |
| G                                      | Card Not Present - Non-3-D Secure  |


## 5. Understanding Payment Processing Fees

Payment Processing Fees depend on a number of characteristics. These characteristics belong to either the merchant or the transaction.

Merchant characteritics include 

* **ID**: identifier of the fee rule within the rule fee dataset
* **card_scheme**: string type. name of the card scheme or network that the fee applies to
* **account_type**: list type. list of account types according to the categorization `Account Type` in this manual
* **capture_delay**: string type. rule that specifies the number of days in which the capture from authorization to settlement needs to happen. Possible values are '3-5' (between 3 and 5 days), '>5' (more than 5 days is possible), '<3' (before 3 days), 'immediate', or 'manual'. The faster the capture to settlement happens, the more expensive it is.
* **monthly_fraud_level**: string type. rule that specifies the fraud levels measured as ratio between monthly total volume and monthly volume notified as fraud. For example '7.7%-8.3%' means that the ratio should be between 7.7 and 8.3 percent. Generally, the payment processors will become more expensive as fraud rate increases.
* **monthly_volume**: string type. rule that specifies the monthly total volume of the merchant. '100k-1m' is between 100.000 (100k) and 1.000.000 (1m). All volumes are specified in euros. Normally merchants with higher volume are able to get cheaper fees from payments processors.
* **merchant_category_code**: list type. integer that specifies the possible merchant category codes, according to the categorization found in this manual in the section `Merchant Category Code`. eg: `[8062, 8011, 8021]`.
* **is_credit**: bool. True if the rule applies for credit transactions. Typically credit transactions are more expensive (higher fee).
* **aci**: list type. string that specifies an array of possible Authorization Characteristics Indicator (ACI) according to the categorization specified in this manual in the section `Authorization Characteristics Indicator`.
* **fixed_amount**: float. Fixed amount of the fee in euros per transaction, for the given rule.
* **rate**: integer. Variable rate to be especified to be multiplied by the transaction value and divided by 10000.
* **intracountry**: bool. True if the transaction is domestic, defined by the fact that the issuer country and the acquiring country are the same. False are for international transactions where the issuer country and acquirer country are different and typically are more expensive.

**Notes**:
* The fee then is provided by `fee = fixed_amount + rate * transaction_value / 10000`.
* Monthly volumes and rates are computed always in natural months (e.g. January, February), starting always in day 1 and ending in the last natural day of the month (i.e. 28 for February, 30 or 31).
* Fixed amount and transaction values are given in the same currency, typically euros.
* If a field is set to null it means that it applies to all possible values of that field. E.g. null value in aci means that the rules applies for all possible values of aci.

The full list of fee rules and values depending on these characteristics can be found in the annexed file `fees.json`. 

###  5.1 Best Practices for Minimizing Transaction Costs


#### 5.1.1 Optimizing Transactions through Local Acquiring

To minimize friction and maximize conversion rates, it is essential to route transactions through local acquirers. Local acquiring refers to the scenario where the issuer country is the same as the acquirer country. This approach can lead to several benefits, including:

- Reduced transaction friction, resulting in higher conversion rates
- Lower fees associated with cross-border transactions

**What is Local Acquiring?**

Local acquiring occurs when a transaction is processed through an acquirer that is located in the same country as the issuer of the card. For example, if a cardholder is located in the United States and makes a purchase from a merchant also located in the United States, the transaction would be considered a local acquiring transaction.

By routing transactions through local acquirers, merchants can reduce the complexity and costs associated with cross-border transactions, ultimately leading to a better user experience and increased conversion rates.

**Benefits of Local Acquiring**

Some of the key benefits of local acquiring include:

- Reduced transaction fees
- Improved conversion rates due to reduced friction
- Enhanced user experience
- Simplified transaction processing

#### 5.1.2. Choosing the right transaction type

**Transaction Processing Options and Fees**

When processing transactions, there are various options available, depending on the type of transaction and the level of authentication required. The Authorization Characteristic Indicator (ACI) provides a standardized way to categorize transactions and determine the best processing method.

**Transaction Processing Methods**

Transactions can be processed in one of several ways, including:

- POS transactions with authentication: This method involves verifying the cardholder's identity through authentication, such as entering a PIN or signature.
- Tokenized transactions: This method involves replacing the cardholder's sensitive information with a token or pseudonym, which can be used to process the transaction.

**Choosing the Right ACI**

When choosing an ACI, consider the following factors:

- Fees: Different ACIs have varying fees associated with them. Choosing the right ACI can help reduce costs, but may also add friction to the transaction process.
- Friction: Some ACIs, such as those that require authentication, may add friction to the transaction process, such as prompting the cardholder to enter a PIN or signature.

**Understanding ACI Codes**

ACI codes are provided in the section `Authorization Characteristics Indicator` and are used to categorize transactions and determine the best processing method. By choosing the right ACI, merchants can optimize their transaction processing and reduce costs.

**Best Practices for Choosing an ACI**

When choosing an ACI, follow these best practices:

- Consider the type of transaction: Different ACIs are suited for different types of transactions, such as POS transactions or e-commerce transactions.
- Consider the level of authentication required: Choose an ACI that provides the required level of authentication, such as authentication or tokenization.
- Consider the fees associated with the ACI: Choose an ACI that balances fees with the level of authentication required and the type of transaction.


# 5.1.3 Processing with Higher Volumes

## Pricing Structure Overview

When processing larger volumes of data, the cost per unit decreases, resulting in a more cost-effective solution. Unlike some pricing models, there is no minimum volume requirement, allowing you to benefit from economies of scale as your needs grow.

## Volume-Based Pricing Curve

The pricing curve is designed to flatten out at higher volumes, ensuring that the cost per unit remains competitive as your volume increases. This means that the more data you process, the lower the cost per unit, allowing you to optimize your budget and achieve a better return on investment.

## Key Benefits

*   No minimum volume requirement, giving you flexibility in your pricing strategy
*   Economies of scale achieved as your volume increases, reducing the cost per unit
*   Competitive pricing at higher volumes, ensuring a better return on investment

#### 5.1.4 Minimizing Fraud-Related Costs

**Understanding the Impact of Fraud Levels**

When processing transactions, it's essential to maintain optimal fraud levels to minimize costs. As fraud levels increase, so do the associated costs. To maximize efficiency and reduce expenses, it's recommended to maintain fraud levels at the lowest possible threshold.

**The Relationship Between Fraud Levels and Costs**

Our pricing model is designed to reflect the increased risk associated with higher fraud levels. As a result, costs will increase in direct proportion to the level of fraud detected. By maintaining optimal fraud levels, you can help reduce these costs and optimize your budget.

**Best Practices for Minimizing Fraud-Related Fees**

For more information on strategies for reducing fraud-related fees, please refer to the `Reducing Fraud-Related Fees` section of this manual. This section provides guidance on how to implement effective anti-fraud measures, monitor transactions, and respond to potential threats.

#### 5.1.5 Avoiding Transaction Downgrades

Transaction downgrades can result in higher processing costs due to less favorable interchange rate tiers. To minimize the risk of downgrades, it is essential to understand the common reasons for downgrades and implement best practices to avoid them.

**Common Reasons for Transaction Downgrades**
- Missing or Incomplete Data Elements: Failing to provide required data elements can lead to downgrades.
- Late Settlement: Settling transactions outside of the designated timeframe can result in downgrades.
- Non-Qualified Transaction Types: Processing transactions that do not meet specific criteria can lead to downgrades.
- Failure to Use AVS or 3D Secure for Card-Not-Present Transactions: Not utilizing enhanced security features for card-not-present transactions can result in downgrades.
- Transaction Size and Volume: Excessive transaction size or volume can lead to downgrades.
- Excessive retrying: Retrying transactions too many times can result in downgrades.

**Best Practices to Avoid Downgrades**

-**Ensure Complete Data Submission**: Provide all required data elements to avoid downgrades.
- **Timely Settlement (within 24 hours)**: Settle transactions within the designated timeframe to avoid downgrades.
- **Use Retry Strategies that Consider Cost and Penalties**: Implement retry strategies that balance cost and penalties to avoid downgrades.
- **Utilize Enhanced Security Features**: Use AVS and 3D Secure for card-not-present transactions to avoid downgrades.
- **Leverage Level 2 and Level 3 Data for B2B Transactions**: Use Level 2 and Level 3 data for B2B transactions to avoid downgrades.
- **Regularly Review and Update Your Systems**: Regularly review and update your systems to ensure compliance with industry standards and avoid downgrades.
- **Train Your Staff**: Train your staff to understand the importance of avoiding downgrades and provide them with the necessary tools and resources to do so.


### 6. PIN Entry Attempt Limits

#### Preventing Unauthorized Access

To maintain the security and integrity of your transactions, we have implemented a PIN entry attempt limit to prevent unauthorized access to your account. This limit is designed to protect you from potential losses due to repeated incorrect PIN attempts.

#### Attempt Limit Details

*   **Maximum Attempts:** Three (3) consecutive incorrect PIN entry attempts are allowed before the card is temporarily blocked.
*   **Temporary Block:** If the attempt limit is reached, your card will be temporarily blocked, and you will be unable to make transactions until the block is lifted.
*   **Unblocking the Card:** To unblock your card or reset your PIN, please contact your issuing bank directly. They will be able to assist you in resolving the issue and reactivating your card for use.
*   **Security Measures:** This limit is in place to prevent unauthorized access to your account and to protect you from potential losses. By limiting the number of incorrect PIN attempts, we can help ensure that your account remains secure and that you can continue to use your card with confidence.

## 7. Reducing Fraud-Related Fees

Fraud is defined as the ratio of fraudulent volume over total volume.

### 7.1 Implementing Proactive Fraud Prevention Strategies

#### Leveraging Advanced Fraud Prevention Tools

To minimize the risk of fraud-related fees, it is essential to implement robust fraud prevention tools. These tools can significantly reduce the likelihood of unauthorized transactions and associated costs. The following measures can be implemented:

*   **Address Verification Service (AVS)**: Verify the billing address of the cardholder to ensure it matches the address on file.
*   **Card Verification Value (CVV) checks**: Validate the CVV code on the card to confirm its authenticity.
*   **3D Secure authentication**: Implement 3D Secure, a payment security protocol that adds an additional layer of authentication for online transactions.
*   **Risk Engine**: Utilize a risk engine that can analyze transaction data and identify suspicious patterns. This can help block attempts that are likely to be fraudulent.

#### Enhancing Transaction Risk Assessment

In addition to the above, a risk engine can be used to determine the nature of the transaction and block attempts that are deemed suspicious. This can be achieved through:

*   **Rules-based engine**: Implement a set of rules that can flag transactions based on specific criteria.
*   **Machine learning engine**: Use machine learning algorithms to analyze transaction data and identify patterns that indicate potential fraud.

### 7.2 Managing Chargebacks Effectively

#### Maintaining a Healthy Chargeback Rate

To avoid penalties and increased costs, it is crucial to maintain a chargeback rate below the desired levels of total transactions. Regularly monitor the chargeback rate and take corrective action when it exceeds acceptable levels.

#### Identifying and Addressing Fraud Rate Drifts

Keep a close eye on the fraud rate drifts and take prompt action when the situation raises to undesired levels. This can help prevent a significant increase in chargebacks and associated costs.

### 7.3 Educating Your Team on Fraud Prevention

#### Training Staff on Best Practices

Train your staff on best practices for handling transactions, including recognizing fraud red flags. This can help them identify and flag suspicious transactions, reducing the risk of fraud-related fees.

### 7.4 Maintaining Compliance with Security Standards

#### Ensuring PCI DSS Compliance

Ensure that your organization complies with the latest Payment Card Industry Data Security Standard (PCI DSS). Failure to comply can result in significant penalties, including:

*   **EUR5,000 to EUR100,000 per month**: Depending on the severity of the non-compliance.
*   **Reputation damage**: Non-compliance can damage your organization's reputation and erode customer trust.

By implementing proactive fraud prevention strategies, managing chargebacks effectively, educating your team, and maintaining compliance with security standards, you can significantly reduce the risk of fraud-related fees and protect your organization's reputation.

## 8. Leveraging Data and Reporting

### 8.1 Unlocking Insights through Transaction Data Analysis

#### Maximizing Cost Savings through Data-Driven Decision Making

Regularly reviewing transaction data is crucial to identifying patterns and opportunities for cost savings. By analyzing your transaction data, you can:

*   **Gain a deeper understanding of your operations**: Identify areas of inefficiency and pinpoint opportunities for improvement.
*   **Optimize your fee structures**: Analyze fee-related data to ensure you're getting the best possible rates.
*   **Enhance your fraud prevention strategies**: Monitor and track key fraud-related metrics to reduce the risk of fraudulent transactions.

### 8.2 Leveraging Reporting Tools for Data-Driven Insights

#### Unlocking Valuable Information with Provided Reporting Tools

To make informed decisions and optimize your operations, it's essential to utilize the provided reporting tools. These tools offer a wealth of information on various aspects of your transactions, including:

*   **Transaction History**: Gain a comprehensive understanding of past transactions, including dates, amounts, and types of transactions.
*   **Fee Structures**: Analyze fee-related data, such as assessment rates, transaction fees, and other charges.
*   **Fraud Metrics**: Monitor and track key fraud-related metrics, including authorization rates, fraud rates, and chargeback rates.

#### Key Performance Indicators (KPIs) to Focus On

To ensure optimal performance and minimize costs, focus on the following key metrics:

*   **Authorization Rate**: Aim for the maximum possible level to maximize successful transactions and minimize rejected transactions.
*   **Fraud Rate**: Strive for the lowest possible level to reduce the risk of fraudulent transactions and associated costs.
*   **Chargeback Rate**: Aim for the lowest possible level to minimize the number of chargebacks and associated fees.

#### Benefits of Tracking Key Metrics

By monitoring and analyzing these key metrics, you can:

*   **Identify areas for improvement**: Pinpoint opportunities to optimize your operations and reduce costs.
*   **Make data-driven decisions**: Base decisions on factual data, rather than intuition or guesswork.
*   **Improve overall performance**: Enhance your authorization rates, reduce fraud rates, and minimize chargeback rates.

By leveraging reporting tools and tracking key metrics, you can gain valuable insights into your transactions and make informed decisions to optimize your operations and minimize costs.

## 9. Appendix

### Glossary

- AVS: Address Verification Service
- CVV: Card Verification Value
- PCI DSS: Payment Card Industry Data Security Standard
- ACI: Authorization Characteristics Indicator

## 10. Contact Information

Merchant Services Support:
- Phone: 1-800-555-1234
- Email: support@paymentprocessor.com
- Website: www.paymentprocessor.com/support

Fraud Prevention Team:
- Phone: 1-800-555-5678
- Email: fraud@paymentprocessor.com

Technical Support:
- Phone: 1-800-555-9876
- Email: tech@paymentprocessor.com

Note: This document is for informational purposes only and does not constitute legal or financial advice. Please consult with your payment processor or a qualified professional for advice specific to your business.

© 2024 Payment Processor, Inc. All rights reserved.