# Description
This Chatbot was created within the context of a small research project for people with
Mild Cognitive Disorder (MCI). The project's goal was to find a way to help people with MCI carry through with
their shopping in an online shop. Towards this goal, we built a Chatbot that would help them
with some tasks that may prove to be difficult for them.

The Chatbot was based on a simple Intent Classification procedure. In short, this Chatbot communicated with the e-shop through an API. When an intent was detected in the user's message, the chatbot would return a response along with an action.
The response would be returned as is to the user and the action would trigger the e-shop to proceed to some actions that would help the user for the specific problem that was detected.

The intents used and the reasons for choosing them are described below:

* **greeting, goodbye, thanks:** Three simple intents in an attempt to make the chatbot be able to make a simple 
and friendly conversation with the user, thus encouraging the user to talk with it.
* **options:** When this intent was detected, the Chatbot would display its capabilities to the user.
* **search_for_product:** When this intent was detected, the Chatbot would trigger the action "search for the product detected in the message". 
* **missing_items:** When this intent was detected, the Chatbot would trigger the action "remind to the user the products he/she has forgotten to buy".
* **confirmation:** This intent was added for reasons related to the eshop's functionality.
* **help:** This intent was added in case the user asked for help in free text form.