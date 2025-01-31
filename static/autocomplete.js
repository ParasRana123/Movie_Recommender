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
        document.querySelector("#movie_list").appendChild(result);
    },
    onSelection: feedback => {
        document.getElementById('autoComplete').value = feedback.selection.value;
    }
});
