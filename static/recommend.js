$(function() {
  // Button will be disabled until we type anything inside the input field
  const source = document.getElementById('autoComplete');
  if (source) {
    const inputHandler = function(e) {
      if(e.target.value.trim() === ""){
        $('.movie-button').attr('disabled', true);
      }
      else{
        $('.movie-button').attr('disabled', false);
      }
    };
    source.addEventListener('input', inputHandler);

    // Support Enter keypress to trigger search
    $('#autoComplete').on('keypress', function(e) {
      if (e.which === 13 || e.keyCode === 13) {
        e.preventDefault();
        $('.movie-button').click();
      }
    });
  }

  $('.movie-button').on('click', function(){
    var title = $('.movie').val();
    if (!title || title.trim() === "") {
      $('.results').css('display','none');
      $('.fail').css('display','block');
    }
    else{
      load_movie_recommendations(title.trim());
    }
  });
});

// will be invoked when clicking on the recommended movies
function recommendcard(e){
  var title = e.getAttribute('title'); 
  if (title) {
    load_movie_recommendations(title);
  }
}

function load_movie_recommendations(title) {
  $("#loader").fadeIn();
  $('.fail').css('display', 'none');
  $('.results').css('display', 'none');

  $.ajax({
    type: 'POST',
    url: '/recommend',
    data: { 'name': title, 'title': title },
    dataType: 'html',
    complete: function() {
      $("#loader").delay(500).fadeOut();
    },
    success: function(response) {
      if (!response || response.indexOf('Sorry! The movie you requested is not in our database') !== -1 || response.indexOf('Movie not found') !== -1) {
        $('.fail').css('display', 'block');
        $('.results').css('display', 'none');
      } else {
        // Hide actor details, underlying movie details, and other static page elements
        $('#actor-main-content').hide();
        $('#movie-main-content').hide();
        $('#page-main-content').hide();
        $('#mycontent').hide();
        $('#streaming-platforms').hide();

        $('.fail').css('display', 'none');
        $('.results').html(response);
        $('.results').css('display', 'block');
        $('#autoComplete').val('');
        $(window).scrollTop(0);
      }
    },
    error: function() {
      alert("Error generating movie recommendations.");
      $('.fail').css('display', 'block');
    }
  });
}
