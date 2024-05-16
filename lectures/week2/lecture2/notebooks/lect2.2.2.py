import turtle
import time

# Set up the screen
screen = turtle.Screen()
screen.title("Turtle While Loop Example")
screen.bgcolor("white")
screen.setup(width=600, height=600)

# Create the turtle
t = turtle.Turtle()
t.shape("turtle")
t.color("green")
t.speed(1)


# Function to check if the turtle has hit the wall
def is_at_wall(turtle):
    x, y = turtle.position()
    screen_width = screen.window_width() / 2
    screen_height = screen.window_height() / 2

    '''
    if abs(x) > screen.window_width()/2 - 150:
        time.sleep(0.2)
    if abs(x) > screen.window_width() / 2 - 50:
        time.sleep(0.5)
    '''

    if abs(x) > screen_width - 20 or abs(y) > screen_height - 20:
        return True
    return False


# Move the turtle forward until it hits the wall
while not is_at_wall(t):
    t.forward(10)

'''
# Move the turtle forward until it hits the wall
at_wall = False

while not at_wall:
    x, y = t.position()
    #print(x,y)
    screen_width = screen.window_width() / 2
    screen_height = screen.window_height() / 2
    if abs(x) > screen_width - 20 or abs(y) > screen_height - 20:
        at_wall = True
    t.forward(10)
'''

# Stop the turtle and display a message
t.write("Hit the wall!", align="center", font=("Arial", 16, "normal"))
t.hideturtle()

# Keep the window open
screen.mainloop()
