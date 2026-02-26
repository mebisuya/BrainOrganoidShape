waitForUser("Draw the first line (cell division plane) and click OK");
getSelectionCoordinates(x1, y1);
waitForUser("Draw the second line (reference surface) and click OK");
getSelectionCoordinates(x2, y2);

// Calculate slopes
m1 = (y1[1] - y1[0]) / (x1[1] - x1[0]);
m2 = (y2[1] - y2[0]) / (x2[1] - x2[0]);

// Calculate the angle
angle = atan(abs((m2 - m1) / (1 + m1 * m2)));
angle = angle * 180 / PI;

// Display the result
print("Angle between the lines: " + angle);