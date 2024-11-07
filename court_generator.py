num_courts = 1

pickleball_courts = []

def create_court():
    # Initialize 44x20 court
    court = [[0 for _ in range(20)] for _ in range(44)]

    # Create net
    for col in range(20):
        court[22][col] = 1

    # Creating kitchen area
    for row in range(15, 22):
        for col in range(20):
            court[row][col] = 2

    for row in range(23, 30):
        for col in range(20):
            court[row][col] = 2

    return court


for _ in range(num_courts):
    new_court = create_court()
    pickleball_courts.append(new_court)

print(pickleball_courts)
    