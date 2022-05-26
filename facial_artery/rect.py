class Rectangle:
    count = 0
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        Rectangle.count += 1
    
    # 인스턴스 메서트    
    def calcArea(self):
        area = self.width * self.height
        return area
    
    # 정적 메서드
    @staticmethod
    def isSquare(rectWidth, rectHeight):
        return rectWidth == rectHeight
    
    # 클래스 메서드
    @classmethod
    def printCount(cls):
        print(cls.count)

# Test
square = Rectangle.isSquare(5,5)
print(square) # true

rect1 = Rectangle(5,5)
rect2 = Rectangle(2,5)
rect1.printCount()  #2

# https://frenchkebab.tistory.com/56

# 정적메소드는 언제 사용하는가?
# https://mygumi.tistory.com/253