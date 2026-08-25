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
    var my_api_key = 'fce0af3409e6113c9b3c75aaf49341bb';
    var title = $('.movie').val();
    if (!title || title.trim() === "") {
      $('.results').css('display','none');
      $('.fail').css('display','block');
    }
    else{
      load_details(my_api_key, title.trim());
    }
  });
});

// will be invoked when clicking on the recommended movies
function recommendcard(e){
  var my_api_key = 'fce0af3409e6113c9b3c75aaf49341bb';
  var title = e.getAttribute('title'); 
  if (title) {
    load_details(my_api_key, title);
  }
}

// get the basic details of the movie from the API (based on the name of the movie)
function load_details(my_api_key, title){
  $("#loader").fadeIn();
  $.ajax({
    type: 'GET',
    url: 'https://api.tmdb.org/3/search/movie?api_key=' + my_api_key + '&query=' + encodeURIComponent(title),

    success: function(movie){
      if(!movie.results || movie.results.length < 1){
        $('.fail').css('display','block');
        $('.results').css('display','none');
        $("#loader").delay(500).fadeOut();
      }
      else{
        $('.fail').css('display','none');
        var movie_id = movie.results[0].id;
        var movie_title = movie.results[0].original_title || movie.results[0].title;
        movie_recs(movie_title, movie_id, my_api_key);
      }
    },
    error: function(){
      alert('Invalid Request or Network Error reaching Movie Database API.');
      $("#loader").delay(500).fadeOut();
    },
  });
}

// passing the movie name to get the similar movies from python's flask
function movie_recs(movie_title, movie_id, my_api_key){
  $.ajax({
    type:'POST',
    url:"/similarity",
    data:{'name': movie_title},
    success: function(recs){
      if(recs == "Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies"){
        $('.fail').css('display','block');
        $('.results').css('display','none');
        $("#loader").delay(500).fadeOut();
      }
      else {
        $('.fail').css('display','none');
        var movie_arr = recs.split('---');
        var arr = [];
        for(const movie in movie_arr){
          if(movie_arr[movie].trim()) {
            arr.push(movie_arr[movie]);
          }
        }
        get_movie_details(movie_id, my_api_key, arr, movie_title);
      }
    },
    error: function(){
      alert("Error retrieving recommendations from server.");
      $("#loader").delay(500).fadeOut();
    },
  }); 
}

// get all the details of the movie using the movie id.
function get_movie_details(movie_id, my_api_key, arr, movie_title) {
  $.ajax({
      type: 'GET',
      url: 'https://api.tmdb.org/3/movie/' + movie_id + '?api_key=' + my_api_key + '&append_to_response=videos,credits',
      success: function(movie_details) {
          let videos = movie_details.videos ? movie_details.videos.results : [];
          let trailer = "";
          let teaser = "";

          if (videos && videos.length > 0) {
              videos.forEach(video => {
                  if (video.type === "Trailer" && video.site === "YouTube") {
                      trailer = `https://www.youtube.com/watch?v=${video.key}`;
                  } else if (video.type === "Teaser" && video.site === "YouTube") {
                      teaser = `https://www.youtube.com/watch?v=${video.key}`;
                  }
              });
          }

          let director = movie_details.credits?.crew?.find(member => member.job === "Director");

          if (director) {
              get_director_details(director.id, my_api_key, movie_id, movie_details, arr, movie_title, trailer, teaser);
          } else {
              let directorDetails = {
                  name: "Unknown",
                  bio: "Biography not available.",
                  place_of_birth: "Unknown",
                  image: "https://via.placeholder.com/300"
              };
              get_watch_providers(movie_id, my_api_key, movie_details, arr, movie_title, trailer, teaser, directorDetails);
          }
      },
      error: function() {
          alert("Error fetching movie details from TMDB API!");
          $("#loader").delay(500).fadeOut();
      },
  });
}

function get_director_details(director_id, my_api_key, movie_id, movie_details, arr, movie_title, trailer, teaser) {
  $.ajax({
      type: 'GET',
      url: `https://api.tmdb.org/3/person/${director_id}?api_key=${my_api_key}`,
      success: function(directorData) {
          let directorDetails = {
              name: directorData.name || "Unknown",
              bio: directorData.biography || "Biography not available.",
              place_of_birth: directorData.place_of_birth || "Unknown",
              image: directorData.profile_path ? `https://image.tmdb.org/t/p/w300${directorData.profile_path}` : "https://via.placeholder.com/300"
          };

          get_watch_providers(movie_id, my_api_key, movie_details, arr, movie_title, trailer, teaser, directorDetails);
      },
      error: function() {
          let directorDetails = {
              name: "Unknown",
              bio: "Biography not available.",
              place_of_birth: "Unknown",
              image: "https://via.placeholder.com/300"
          };
          get_watch_providers(movie_id, my_api_key, movie_details, arr, movie_title, trailer, teaser, directorDetails);
      }
  });
}

// Fetch watch providers (streaming platforms)
function get_watch_providers(movie_id, my_api_key, movie_details, arr, movie_title, trailer, teaser, directorDetails) {
  $.ajax({
      type: 'GET',
      url: `https://api.tmdb.org/3/movie/${movie_id}/watch/providers?api_key=${my_api_key}`,
      success: function(providerData) {
          let providerNames = [];
          let providerLogos = [];

          if (providerData.results && providerData.results.IN && providerData.results.IN.flatrate) {
              let providers = providerData.results.IN.flatrate;
              providerNames = providers.map(provider => provider.provider_name);
              providerLogos = providers.map(provider => provider.logo_path ? `https://image.tmdb.org/t/p/w200${provider.logo_path}` : "");
          } else if (providerData.results && providerData.results.US && providerData.results.US.flatrate) {
              let providers = providerData.results.US.flatrate;
              providerNames = providers.map(provider => provider.provider_name);
              providerLogos = providers.map(provider => provider.logo_path ? `https://image.tmdb.org/t/p/w200${provider.logo_path}` : "");
          }

          show_details(movie_details, arr, movie_title, my_api_key, movie_id, trailer, teaser, providerNames, providerLogos, directorDetails);
      },
      error: function() {
          show_details(movie_details, arr, movie_title, my_api_key, movie_id, trailer, teaser, [], [], directorDetails);
      }
  });
}

// passing all the details to python's flask for displaying and scraping the movie reviews using imdb id
function show_details(movie_details, arr, movie_title, my_api_key, movie_id, trailer, teaser, providerNames, providerLogos, directorDetails){
  var imdb_id = movie_details.imdb_id || "";
  var poster = movie_details.poster_path ? 'https://image.tmdb.org/t/p/original' + movie_details.poster_path : 'https://via.placeholder.com/500x750';
  var backdrop = movie_details.backdrop_path ? 'https://image.tmdb.org/t/p/original' + movie_details.backdrop_path : poster;
  var overview = movie_details.overview || "No overview available.";
  var genres = movie_details.genres || [];
  var rating = movie_details.vote_average || 0;
  var vote_count = movie_details.vote_count || 0;
  var release_date = movie_details.release_date ? new Date(movie_details.release_date) : new Date();
  var runtime = parseInt(movie_details.runtime) || 0;
  var status = movie_details.status || "Released";
  var budget = movie_details.budget ? movie_details.budget.toLocaleString() : "N/A";
  var revenue = movie_details.revenue ? movie_details.revenue.toLocaleString() : "N/A";
  var original_language = movie_details.original_language ? movie_details.original_language.toUpperCase() : "EN";

  var genre_list = [];
  for (var genre in genres){
    if (genres[genre] && genres[genre].name) {
      genre_list.push(genres[genre].name);
    }
  }
  var my_genre = genre_list.join(", ");
  if(runtime > 0) {
    if(runtime % 60 == 0){
      runtime = Math.floor(runtime/60) + " hour(s)";
    }
    else {
      runtime = Math.floor(runtime/60) + " hour(s) " + (runtime%60) + " min(s)";
    }
  } else {
    runtime = "N/A";
  }

  var arr_poster = get_movie_posters(arr, my_api_key);
  var movie_cast = get_movie_cast(movie_id, my_api_key);
  var ind_cast = get_individual_cast(movie_cast, my_api_key);
  
  var details = {
    'title': movie_title,
    'cast_ids': JSON.stringify(movie_cast.cast_ids),
    'cast_names': JSON.stringify(movie_cast.cast_names),
    'cast_chars': JSON.stringify(movie_cast.cast_chars),
    'cast_profiles': JSON.stringify(movie_cast.cast_profiles),
    'cast_bdays': JSON.stringify(ind_cast.cast_bdays),
    'cast_bios': JSON.stringify(ind_cast.cast_bios),
    'cast_places': JSON.stringify(ind_cast.cast_places),
    'imdb_id': imdb_id,
    'poster': poster,
    'backdrop': backdrop,
    'genres': my_genre,
    'overview': overview,
    'rating': rating,
    'vote_count': vote_count.toLocaleString(),
    'release_date': release_date.toDateString().split(' ').slice(1).join(' '),
    'runtime': runtime,
    'status': status,
    'trailer': trailer,
    'teaser': teaser,
    'watch_providers': JSON.stringify(providerNames),
    'watch_provider_logos': JSON.stringify(providerLogos),
    'rec_movies': JSON.stringify(arr),
    'rec_posters': JSON.stringify(arr_poster),
    'budget': budget,
    'revenue': revenue,
    'original_language': original_language,
    'director_name': directorDetails.name || "Unknown",
    'director_bio': directorDetails.bio || "Biography not available.",
    'director_place_of_birth': directorDetails.place_of_birth || "Unknown",
    'director_image': directorDetails.image || "https://via.placeholder.com/300"
  };

  $.ajax({
    type:'POST',
    data: details,
    url: "/recommend",
    dataType: 'html',
    complete: function(){
      $("#loader").delay(500).fadeOut();
    },
    success: function(response) {
      $('.results').html(response);
      $('.results').css('display', 'block');
      $('#autoComplete').val('');
      $(window).scrollTop(0);
    },
    error: function() {
      alert("Error generating recommendation view.");
    }
  });
}

// get the details of individual cast
function get_individual_cast(movie_cast, my_api_key) {
  var cast_bdays = [];
  var cast_bios = [];
  var cast_places = [];
  for(var cast_id in movie_cast.cast_ids){
    $.ajax({
      type:'GET',
      url:'https://api.tmdb.org/3/person/' + movie_cast.cast_ids[cast_id] + '?api_key=' + my_api_key,
      async: false,
      success: function(cast_details){
        var bday = cast_details.birthday ? (new Date(cast_details.birthday)).toDateString().split(' ').slice(1).join(' ') : 'Unknown';
        cast_bdays.push(bday);
        cast_bios.push(cast_details.biography || 'Biography not available.');
        cast_places.push(cast_details.place_of_birth || 'Unknown');
      },
      error: function() {
        cast_bdays.push('Unknown');
        cast_bios.push('Biography not available.');
        cast_places.push('Unknown');
      }
    });
  }
  return {cast_bdays: cast_bdays, cast_bios: cast_bios, cast_places: cast_places};
}

// getting the details of the cast for the requested movie
function get_movie_cast(movie_id, my_api_key){
  var cast_ids = [];
  var cast_names = [];
  var cast_chars = [];
  var cast_profiles = [];

  $.ajax({
    type: 'GET',
    url: "https://api.tmdb.org/3/movie/" + movie_id + "/credits?api_key=" + my_api_key,
    async: false,
    success: function(my_movie){
      if (my_movie.cast && my_movie.cast.length > 0) {
        var top_count = Math.min(my_movie.cast.length, 10);
        for(var i = 0; i < top_count; i++){
          cast_ids.push(my_movie.cast[i].id);
          cast_names.push(my_movie.cast[i].name);
          cast_chars.push(my_movie.cast[i].character || "");
          var profile = my_movie.cast[i].profile_path 
            ? "https://image.tmdb.org/t/p/original" + my_movie.cast[i].profile_path 
            : "https://via.placeholder.com/240x360?text=No+Image";
          cast_profiles.push(profile);
        }
      }
    },
    error: function(){
      console.warn("Could not fetch cast credits for movie_id: " + movie_id);
    }
  });

  return {cast_ids: cast_ids, cast_names: cast_names, cast_chars: cast_chars, cast_profiles: cast_profiles};
}

// getting posters for all the recommended movies
function get_movie_posters(arr, my_api_key){
  var arr_poster_list = [];
  for(var m in arr) {
    $.ajax({
      type:'GET',
      url:'https://api.tmdb.org/3/search/movie?api_key=' + my_api_key + '&query=' + encodeURIComponent(arr[m]),
      async: false,
      success: function(m_data){
        if (m_data.results && m_data.results.length > 0 && m_data.results[0].poster_path) {
          arr_poster_list.push('https://image.tmdb.org/t/p/original' + m_data.results[0].poster_path);
        } else {
          arr_poster_list.push('https://via.placeholder.com/240x360?text=No+Poster');
        }
      },
      error: function(){
        arr_poster_list.push('https://via.placeholder.com/240x360?text=No+Poster');
      },
    });
  }
  return arr_poster_list;
}
