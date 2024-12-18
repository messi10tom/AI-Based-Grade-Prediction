class specs:
    batch_size = 32
    epochs = 1000
    learning_rate = 0.001
    input_size = 30
    test_size = 0.2
    data_columns_to_encode = ['sex',
                          'address',
                          'famsize',
                          'Pstatus',
                          'Mjob',
                          'Fjob',
                          'reason',
                          'schoolsup',
                          'famsup',
                          'paid',
                          'activities',
                          'nursery',
                          'higher',
                          'internet',
                          'romantic',
                          ]
    norm_specs = {
    "mean_dict": {
        "sex": 0.4339080459770115,
        "age": 16.726053639846743,
        "address": 0.27298850574712646,
        "famsize": 0.29310344827586204,
        "Pstatus": 0.8840996168582376,
        "Medu": 2.603448275862069,
        "Fedu": 2.3879310344827585,
        "Mjob": 2.0277777777777777,
        "Fjob": 1.4741379310344827,
        "reason": 1.3103448275862069,
        "traveltime": 1.5229885057471264,
        "studytime": 1.9703065134099618,
        "failures": 0.26436781609195403,
        "schoolsup": 0.8860153256704981,
        "famsup": 0.6130268199233716,
        "paid": 0.210727969348659,
        "activities": 0.4942528735632184,
        "nursery": 0.20019157088122605,
        "higher": 0.08524904214559387,
        "internet": 0.7921455938697318,
        "romantic": 0.3553639846743295,
        "famrel": 3.935823754789272,
        "freetime": 3.2011494252873565,
        "goout": 3.1561302681992336,
        "Dalc": 1.4942528735632183,
        "Walc": 2.2844827586206895,
        "health": 3.543103448275862,
        "absences": 4.434865900383142,
        "G1": 11.21360153256705,
        "G2": 11.246168582375478,
        "G3": 11.341954022988507
    },
    "std_dict": {
        "sex": 0.4956126043730357,
        "age": 1.2393806932778537,
        "address": 0.4454949847945289,
        "famsize": 0.4551854752566926,
        "Pstatus": 0.3201054268976943,
        "Medu": 1.124367675087003,
        "Fedu": 1.0994111673000175,
        "Mjob": 1.2442182163006505,
        "Fjob": 0.8992875661149216,
        "reason": 1.229992475704359,
        "traveltime": 0.7313768807053818,
        "studytime": 0.8339535059647104,
        "failures": 0.6558274604000436,
        "schoolsup": 0.3177926499268026,
        "famsup": 0.48705742780293343,
        "paid": 0.4078255659995455,
        "activities": 0.49996696944670216,
        "nursery": 0.4001436065081299,
        "higher": 0.27925193456599834,
        "internet": 0.4057720443580382,
        "romantic": 0.478623466903486,
        "famrel": 0.9329536335464842,
        "freetime": 1.031012698548682,
        "goout": 1.1520225285999592,
        "Dalc": 0.9112775321781297,
        "Walc": 1.2844891871576642,
        "health": 1.4240209195572826,
        "absences": 6.207041706358143,
        "G1": 2.9819647631866038,
        "G2": 3.283497381153174,
        "G3": 3.8629444050896353
    }
}
    encoding_vocab = {
    "sex": [
        "F",
        "M"
    ],
    "address": [
        "U",
        "R"
    ],
    "famsize": [
        "GT3",
        "LE3"
    ],
    "Pstatus": [
        "A",
        "T"
    ],
    "Mjob": [
        "at_home",
        "health",
        "other",
        "services",
        "teacher"
    ],
    "Fjob": [
        "teacher",
        "other",
        "services",
        "health",
        "at_home"
    ],
    "reason": [
        "course",
        "other",
        "home",
        "reputation"
    ],
    "schoolsup": [
        "yes",
        "no"
    ],
    "famsup": [
        "no",
        "yes"
    ],
    "paid": [
        "no",
        "yes"
    ],
    "activities": [
        "no",
        "yes"
    ],
    "nursery": [
        "yes",
        "no"
    ],
    "higher": [
        "yes",
        "no"
    ],
    "internet": [
        "no",
        "yes"
    ],
    "romantic": [
        "no",
        "yes"
    ]
}
    