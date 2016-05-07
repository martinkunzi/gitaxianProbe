The goal of this part of the project is to recognise a MTG extension. 

The script will be given a magic card in a fixed format (223 x 310)  as an argument

	-- example : 
		python extensionIdentifier.py senTriplets.jpg 
		
He'll look for the extension symbole zone (middle right of the image), take the extension symbol, and compare it to the symbol library
He might look elsewhere if the card is one who has a misplaced extension symbol (e.g. full art lands)
Normally, the format of the card will be taken care of by the pretreatment part of the program