import turtle

# Set up the screen
screen = turtle.Screen()
screen.title("Turtle While Loop Example")
screen.bgcolor("white")
screen.setup(width=600, height=600)


'''
lebron = turtle.Turtle()

lebron.forward(50)
lebron.left(90)
lebron.forward(50)
lebron.left(90)
lebron.forward(50)
lebron.left(90)
lebron.forward(50)
lebron.left(90)

turtle.exitonclick()
'''



'''
def draw_square(t, size):
    t.forward(size)
    t.left(90)
    t.forward(size)
    t.left(90)
    t.forward(size)
    t.left(90)
    t.forward(size)
    t.left(90)


lebron = turtle.Turtle()
draw_square(lebron, 20)
draw_square(lebron, 40)
draw_square(lebron, 55)

turtle.exitonclick()
'''



'''
def drawSquare(t, size):  
    t.forward(size)
    t.left(90)
    t.forward(size)
    t.left(90)
    t.forward(size)
    t.left(90)
    t.forward(size)
    t.left(90)

lebron = turtle.Turtle()
lebron.color('red')
lebron.begin_fill()
drawSquare(lebron, 25)
lebron.end_fill()
drawSquare(lebron, 40)
drawSquare(lebron, 55)

turtle.exitonclick()
'''


'''
lebron = turtle.Turtle()
lebron.color('red')
lebron.speed(10)
lebron.begin_fill()
back_to_beginning = False

while not back_to_beginning:
    lebron.forward(200)
    lebron.left(170)
    back_to_beginning = (abs(lebron.pos()) < 1)
    print(lebron.pos())
    print('Final:', abs(lebron.pos()))

print('Outside:', abs(lebron.pos()))

lebron.end_fill()
turtle.exitonclick()
'''