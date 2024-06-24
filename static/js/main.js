$(document).ready(function () {
  $("#upload-form").submit(function (e) {
    e.preventDefault();
    var formData = new FormData($(this)[0]);

    var reader = new FileReader();
    reader.onload = function (e) {
      $("#chat-box").append(
        '<div class="message user-message"><img src="' +
          e.target.result +
          '" width="100"></div>'
      );
      $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
    };
    reader.readAsDataURL($("#file-input")[0].files[0]);

    $.ajax({
      type: "POST",
      url: "/predict",
      data: formData,
      contentType: false,
      cache: false,
      processData: false,
      success: function (response) {
        $("#chat-box").append(
          '<div class="message bot-message">Result: ' +
            response.class +
            "</div>"
        );
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
      },
      error: function () {
        $("#chat-box").append(
          '<div class="message bot-message">An error occurred. Please try again.</div>'
        );
        $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
      },
    });
  });
});
