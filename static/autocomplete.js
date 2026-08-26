(function() {
    function setupAutoComplete(filmsList) {
        const inputElement = document.querySelector("#autoComplete");
        if (!inputElement || typeof autoComplete === 'undefined') return;

        // Prevent double initialization
        if (inputElement.dataset.autocompleteInitialized === "true") return;
        inputElement.dataset.autocompleteInitialized = "true";

        new autoComplete({
            data: {
                src: filmsList,
                cache: true
            },
            selector: "#autoComplete",
            threshold: 2,
            debounce: 200,
            searchEngine: "strict",
            resultsList: {
                render: true,
                container: source => {
                    source.setAttribute("id", "movie_list");
                },
                destination: document.querySelector("#autoComplete"),
                position: "afterend",
                element: "ul"
            },
            maxResults: 10,
            highlight: true,
            resultItem: {
                content: (data, source) => {
                    source.innerHTML = data.match;
                },
                element: "li"
            },
            noResults: () => {
                const result = document.createElement("li");
                result.setAttribute("class", "no_result");
                result.setAttribute("tabindex", "1");
                result.innerHTML = "No Results Found";
                const movieList = document.querySelector("#movie_list");
                if (movieList) {
                    movieList.appendChild(result);
                }
            },
            onSelection: feedback => {
                const input = document.getElementById('autoComplete');
                if (input) {
                    input.value = feedback.selection.value;
                }
                const btn = document.querySelector('.movie-button');
                if (btn) {
                    btn.removeAttribute('disabled');
                    btn.click();
                }
            }
        });
    }

    function init() {
        if (!document.querySelector("#autoComplete")) return;

        if (typeof window.films !== 'undefined' && Array.isArray(window.films) && window.films.length > 0) {
            setupAutoComplete(window.films);
        } else {
            // Asynchronously fetch movie suggestions from server
            fetch('/suggestions')
                .then(response => {
                    if (!response.ok) throw new Error("Failed to load suggestions");
                    return response.json();
                })
                .then(data => {
                    window.films = data;
                    setupAutoComplete(data);
                })
                .catch(err => {
                    console.warn("Could not load suggestions list:", err);
                });
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();