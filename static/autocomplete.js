if (document.querySelector("#autoComplete") && typeof films !== 'undefined') {
    new autoComplete({
        data: {
            src: films,
            cache: true
        },
        selector: "#autoComplete",
        threshold: 2,
        debounce: 300, // Lower the debounce to make it more responsive
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
            result.innerHTML = "No Results";
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
            }
        }
    });
}