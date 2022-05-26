

def standardize(X):
    X_stand = []
    batchsize = {
        4: 0.16,
        8: 0.32,
        16: 0.48,
        32: 0.64,
        64: 0.80,
        128: 0.96
    }
    cpus = {
        1: 0.2,
        2: 0.4,
        3: 0.6,
        4: 0.8,
        5: 1.0
    }
    gpumem = {
        0.8: 0.25,
        1.2: 0.5,
        1.6: 0.75,
        2.4: 1.0
    }
    gpupower = {
        50: 0.33,
        75: 0.66,
        100: 0.99
    }
    gputype = {
        1: 0.25,
        2: 0.5,
        3: 0.75,
        4: 1.0
    }

    # for i in range(0, len(X)):
    #     for j in range(0, len(X[i])):
    #         if i == 0:
    #             X[i][j][0] = batchsize.get(X[i][j][0], None)
    #             X[i][j][1] = gputype.get(X[i][j][1], None)
    #         elif i == 1:
    #             X[i][j][0] = cpus.get(X[i][j][0], None)
    #             X[i][j][1] = gputype.get(X[i][j][1], None)
    #         elif i == 2:
    #             X[i][j][0] = cpus.get(X[i][j][0], None)
    #             X[i][j][1] = gpupower.get(X[i][j][1], None)
    #         elif i == 3:
    #             X[i][j][0] = cpus.get(X[i][j][0], None)
    #             X[i][j][1] = gpumem.get(X[i][j][1], None)

    X[0] = batchsize.get(X[0], None)
    X[1] = cpus.get(X[1], None)
    X[2] = gpumem.get(X[2], None)
    X[3] = gpupower.get(X[3], None)
    X[4] = gputype.get(X[4], None)

    return X

def unstandardize(X):
    for i in range(0, len(X)):
        if i == 0:
            if X[i] <= 0.24:
                X[i] = 4
            elif 0.24 < X[i] <= 0.4:
                X[i] = 8
            elif 0.4 < X[i] <= 0.56:
                X[i] = 16
            elif 0.56 < X[i] <= 0.72:
                X[i] = 32
            elif 0.72 < X[i] <= 0.88:
                X[i] = 64
            elif 0.88 < X[i]:
                X[i] = 128
        elif i == 1:
            if X[i] <= 0.3:
                X[i] = 1
            elif 0.3 < X[i] <= 0.5:
                X[i] = 2
            elif 0.5 < X[i] <= 0.7:
                X[i] = 3
            elif 0.7 < X[i] <= 0.9:
                X[i] = 4
            elif 0.9 < X[i]:
                X[i] = 5
        elif i == 2:
            if X[i] <= 0.375:
                X[i] = 0.8
            elif 0.375 < X[i] <= 0.625:
                X[i] = 1.2
            elif 0.625 < X[i] <= 0.875:
                X[i] = 1.6
            elif 0.875 < X[i]:
                X[i] = 2.4
        elif i == 3:
            if X[i] <= 0.495:
                X[i] = 50
            elif 0.495 < X[i] <= 0.825:
                X[i] = 75
            elif 0.825 < X[i]:
                X[i] = 100
        elif i == 4:
            if X[i] <= 0.375:
                X[i] = 1
            elif 0.375 < X[i] <= 0.625:
                X[i] = 2
            elif 0.625 < X[i] <= 0.875:
                X[i] = 3
            elif 0.875 < X[i]:
                X[i] = 4

    return X



