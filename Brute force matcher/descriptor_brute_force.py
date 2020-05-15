

def bruteForceHamming(descriptor1,descriptor2):
    value = []
    for i in range(32):
        value.append(descriptor1[i] ^ descriptor2[i])
    output = 0
    for i in value:
        binary = bin(i)[2:]
        output = output + binary.count('1')
    return output


f= open("Brute_force_descriptor_test.txt", "r")

i = 0
while(i < 423):
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()

    line1 = line1[1:len(line1)-2]
    line2 = line2[1:len(line2)-2]
    line3 = line3[:len(line3)-1]

    descriptor1 = list(map(int,line1.split(",")))
    descriptor2 = list(map(int,line2.split(",")))
    distance    = int(line3)

    print(str(i + 1) + "th iteration: ", end=" ")
    print(distance == bruteForceHamming(descriptor1,descriptor2), end=" ")
    print(distance)
    i = i + 1
