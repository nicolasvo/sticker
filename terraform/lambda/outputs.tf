output "url_bot" {
  value = "${aws_apigatewayv2_stage.api_gw_bot.invoke_url}/sticker-bot"
}

output "url_bot_down" {
  value = "${aws_apigatewayv2_stage.api_gw_bot_down.invoke_url}/sticker-bot-down"
}
