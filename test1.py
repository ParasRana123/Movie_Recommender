from flask import Flask , render_template
import logging
app = Flask(__name__)
import csv

@app.route("/actor/<actor_id>")
def actor_details(actor_id):
    try:
        global cast_details  # Ensure the dictionary is accessible
        movies = []

        logging.debug(f"Fetching details for Actor ID: {actor_id}")

        # 🚨 Debugging Step: Log if cast_details is empty
        if not cast_details:
            logging.error("cast_details dictionary is EMPTY! Check data loading.")
            return render_template("error.html", message="Cast details not found.")

        # 🚨 Debugging Step: Log all actor IDs present
        available_actor_ids = [str(details[0]) for details in cast_details.values()]
        logging.debug(f"Available Actor IDs in cast_details: {available_actor_ids}")

        # Ensure actor_id is compared as a string (Flask request params are always strings)
        actor_id = str(actor_id)

        # 🔍 Map actor_id to actor_name
        actor_name = None
        for name, details in cast_details.items():
            logging.debug(f"Checking {name}: {details}")  # Log each entry
            if str(details[0]) == actor_id:  # Ensure correct type comparison
                actor_name = name
                break

        # 🔴 If actor_id is not found in cast_details, log error and return
        if not actor_name:
            logging.error(f"Actor ID {actor_id} not found in cast details dictionary!")
            return render_template("error.html", message=f"Actor ID {actor_id} not found in the database.")

        logging.debug(f"Matched Actor: {actor_name}")

        # 🎥 Fetch movies from actors1.csv
        try:
            with open('actors1.csv', mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    logging.debug(f"Checking CSV Row: {row}")  # Debug CSV parsing
                    if row['actor_name'].strip().lower() == actor_name.strip().lower():
                        movies.append(row['movie_title'])
        except FileNotFoundError:
            logging.error("actors1.csv file not found!")
            return render_template("error.html", message="Actor database not found.")

        logging.debug(f"Movies for {actor_name}: {movies}")

        # 🔴 Ensure actor details exist in cast_details
        if actor_name not in cast_details:
            logging.error(f"Actor {actor_name} not found in cast_details!")
            return render_template("error.html", message="Actor details not found.")

        details = cast_details[actor_name]

        # 🔴 Validate expected details structure
        if len(details) < 5:
            logging.error(f"Incomplete details for {actor_name}: {details}")
            return render_template("error.html", message="Actor details are incomplete.")

        # 📌 Construct actor details dictionary
        actor = {
            "name": actor_name,
            "profile": details[1] if details[1] else "https://via.placeholder.com/150",
            "birthday": details[2] if details[2] else "Unknown",
            "birth_place": details[3] if details[3] else "Unknown",
            "biography": details[4] if details[4] else "No biography available."
        }

        return render_template("actor.html", actor=actor, movies=movies)

    except Exception as e:
        logging.error(f"Error loading actor details: {e}", exc_info=True)
        return render_template("error.html", message="An error occurred while fetching actor details.")
    
actor_details(69055)
    

if __name__ == "__main__":
    app.run(debug=True)