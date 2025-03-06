function twoDPFplot(xData, yData, color)
    % Combine 'xData' and 'yData' into a 2xN matrix
    points = [xData'; yData'];
    
    % Sort the points by x-coordinate
    [sortedX, sortOrder] = sort(points(1,:));
    sortedY = points(2, sortOrder);
    
    % Plot the sorted points with the specified color
    plot(sortedX, sortedY, '-*', 'Color', color);
end
