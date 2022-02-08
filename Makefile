bot_api_token =
url_up =
url_down =
# url_ngrok =

up:
	curl https://api.telegram.org/bot${bot_api_token}/setWebHook?url=${url_up}

down:
	curl https://api.telegram.org/bot${bot_api_token}/setWebHook?url=${url_down}

dev:
	curl https://api.telegram.org/bot${bot_api_token}/setWebHook?url=${url_ngrok}/functions/function/invocations

remove:
	curl https://api.telegram.org/bot${bot_api_token}/deleteWebhook

update:
	curl https://api.telegram.org/bot${bot_api_token}/getUpdates
