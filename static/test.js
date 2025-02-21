// get all the details of the movie using the movie id.
function get_movie_details(movie_id, my_api_key, arr, movie_title) {
    $.ajax({
        type: 'GET',
        url: 'https://api.themoviedb.org/3/movie/' + movie_id + '?api_key=' + my_api_key + '&append_to_response=videos,credits',
        success: function(movie_details) {
            let videos = movie_details.videos.results;
            let trailer = "";
            let teaser = "";
  
            if (videos.length > 0) {
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
            alert("API Error!");
            $("#loader").delay(500).fadeOut();
        },
    });
  }
  
  function get_director_details(director_id, my_api_key, movie_id, movie_details, arr, movie_title, trailer, teaser) {
    $.ajax({
        type: 'GET',
        url: `https://api.themoviedb.org/3/person/${director_id}?api_key=${my_api_key}`,
        success: function(directorData) {
            let directorDetails = {
                name: directorData.name,
                bio: directorData.biography || "Biography not available.",
                place_of_birth: directorData.place_of_birth || "Unknown",
                image: directorData.profile_path ? `https://image.tmdb.org/t/p/w300${directorData.profile_path}` : "https://via.placeholder.com/300"
            };
  
            get_watch_providers(movie_id, my_api_key, movie_details, arr, movie_title, trailer, teaser, directorDetails);
        },
        error: function() {
            console.log("Error fetching director details.");
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
        url: `https://api.themoviedb.org/3/movie/${movie_id}/watch/providers?api_key=${my_api_key}`,
        success: function(providerData) {
            let providerNames = [];
            let providerLogos = [];
  
            if (providerData.results && providerData.results.IN && providerData.results.IN.flatrate) {  // Change 'IN' for your country
                let providers = providerData.results.IN.flatrate;
                providerNames = providers.map(provider => provider.provider_name);
                providerLogos = providers.map(provider => provider.logo_path ? `https://image.tmdb.org/t/p/w200${provider.logo_path}` : "");
            }
  
            show_details(movie_details, arr, movie_title, my_api_key, movie_id, trailer, teaser, providerNames, providerLogos, directorDetails);
        },
        error: function() {
            console.log("Error fetching watch providers.");
            show_details(movie_details, arr, movie_title, my_api_key, movie_id, trailer, teaser, [], [], directorDetails);
        }
    });
  }
  
  // Passing all the details to Python's Flask for displaying and scraping the movie reviews using IMDb ID
  function show_details(movie_details, arr, movie_title, my_api_key, movie_id, trailer, teaser, providerNames, providerLogos, directorDetails) {
    var imdb_id = movie_details.imdb_id;
    var poster = 'https://image.tmdb.org/t/p/original' + movie_details.poster_path;
    var backdrop = 'https://image.tmdb.org/t/p/original' + movie_details.backdrop_path;
    var overview = movie_details.overview;
    var genres = movie_details.genres;
    var rating = movie_details.vote_average;
    var vote_count = movie_details.vote_count;
    var release_date = new Date(movie_details.release_date);
    var runtime = parseInt(movie_details.runtime);
    var status = movie_details.status;
    var budget = movie_details.budget.toLocaleString();
    var revenue = movie_details.revenue.toLocaleString();
    var original_language = movie_details.original_language.toUpperCase();
  
    var genre_list = genres.map(genre => genre.name);
    var my_genre = genre_list.join(", ");
  
    if (runtime % 60 == 0) {
        runtime = Math.floor(runtime / 60) + " hour(s)";
    } else {
        runtime = Math.floor(runtime / 60) + " hour(s) " + (runtime % 60) + " min(s)";
    }
  
    arr_poster = get_movie_posters(arr, my_api_key);
    movie_cast = get_movie_cast(movie_id, my_api_key);
    ind_cast = get_individual_cast(movie_cast, my_api_key);
  
    details = {
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
        'director_name': directorDetails.name,
        'director_bio': directorDetails.bio,
        'director_place_of_birth': directorDetails.place_of_birth,
        'director_image': directorDetails.image
    };
  
    $.ajax({
        type: 'POST',
        data: details,
        url: "/recommend",
        dataType: 'html',
        complete: function() {
            $("#loader").delay(500).fadeOut();
        },
        success: function(response) {
            $('.results').html(response);
            $('#autoComplete').val('');
            $(window).scrollTop(0);
        }
    });
  }
  