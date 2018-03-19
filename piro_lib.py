import math

def fit_division_line(center_x,center_y, corners):
    const = 100
    for angle in range(0,360,1):
        second_point_y = center_y + const*math.sin(angle)
        second_point_x = center_x + const*math.cos(angle)
        if center_x == second_point_x and center_y == second_point_y or (center_x - second_point_x) == 0:
            continue
        a_val = ((center_y - second_point_y)/(center_x - second_point_x))
        b_val = center_y - a_val*center_x

        over = 0
        under = []
        corner_count = len(corners)
        for c in corners:
            x,y = c.ravel()

            if y > x*a_val + b_val:
                over += 1
            else:
                under.append(c)
        if corner_count - over == 2:
            print((a_val, b_val))
            return (a_val, b_val, under)

    return None

#fit_division_line(0,0, np.array([[-2,1],[-3,1],[-2,1],[-1,2],[0,1],[1,2],[2,1],[2,0]]))