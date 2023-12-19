## Project Structure:
* Actor - a general class, supports several different types of operating unit, including direct user control
* Environment - The Gym environment for the game. 
* Map - The class handling the key details of the map that the actors are acting in.
* Rendering Tools - Utilized by environment if rendering mode is turned on.
* Main - Where everything comes together. Actor and environment are initialized here.