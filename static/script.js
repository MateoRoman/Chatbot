$(document).ready(function() {
    $("#messageArea").on("submit", function(event) {
        const date = new Date();
        const hour = date.getHours();
        const minute = date.getMinutes();
        const str_time = hour + ":" + minute;
        var rawText = $("#user-input").val();

        var userHtml = '<div class="d-flex justify-content-end mb-4"><div class="msg_cotainer_send">' + rawText + '<span class="msg_time_send">' + str_time + '</span></div><div class="img_cont_msg"><img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" class="rounded-circle user_img_msg"></div></div>';
        
        $("#user-input").val("");
        $("#chat-box").append(userHtml);

        $.ajax({
            data: {
                msg: rawText,    
            },
            type: "POST",
            url: "/get",
            dataType: "json",  // Especifica que esperas un JSON
        }).done(function(data) {
            var botHtml = '<div class="d-flex justify-content-start mb-4"><div class="img_cont_msg"><img src="https://i.ibb.co/fSNP7Rz/icons8-chatgpt-512.png" class="rounded-circle user_img_msg"></div><div class="msg_cotainer">' + data.answer + '<span class="msg_time">' + str_time + '</span></div></div>';
            $("#chat-box").append($.parseHTML(botHtml));
        }).fail(function(xhr, status, error) {
            console.error("Error:", error);
        });

        event.preventDefault();
    });
});
