
import os
import requests
import json

SLACK_WEBHOOK = "https://hooks.slack.com/services/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"
#SLACK_WEBHOOK= os.environ.get("SLACK_WEBHOOK")


def send_message(messages, channel="notifications", username="omkarreddy2008"):
    """
    :param messages: list of texts
    :param channel: name of slack channel
    :param username: username of the bot
    """
    data = {
            "username": username,
            "channel": channel
        }
    data['text'] = '\n'.join(messages)
    response = requests.post(SLACK_WEBHOOK, json.dumps(data))

    print('Response: ' + str(response.text))
    print('Response code: ' + str(response.status_code))


def send_dict(dictionary, channel="general", username="omkarreddy23"):
    """
    :param dictionary: a dictionary (key, value)
    :param channel: name of slack channel
    :param username: username of the bot
    """
    data = {
            "username": username,
            "channel": channel
        }
    values = []
    for k, v in dictionary.items():
        values.append(str(k) + ": " + str(v))
    
    data["text"] = "\n".join(values)
    requests.post(SLACK_WEBHOOK, json.dumps(data))

#send_message(["Hi"])
#send_message(["", "Hi Omkar,", "This is your notification bot testing.","Regards,", "Bot"])
