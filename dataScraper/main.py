import csv, copy, requests

url = "https://static.usafacts.org/public/data/covid-19/covid_confirmed_usafacts.csv"
r = requests.get(url, allow_redirects=True)

open('cases.csv', 'wb').write(r.content)

with open('cases.csv', 'r') as csv_file:
    d_reader = csv.DictReader(csv_file)
    csv_reader = csv.reader(csv_file)

    headers = d_reader.fieldnames
    next(csv_reader)

    for line in csv_reader:
        if line[1] == "Putnam County " and line[2] == "NY":
            newLine = copy.deepcopy(line)
            for i in range(5, len(line)):
                if int(line[i]) > 0:
                    hold = int(newLine[i])
                    hold = int(line[i]) - int(line[i - 1])
                    newLine[i] = str(hold)
            with open('output.csv', mode='w') as output:
                writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                writer.writerow(headers)
                writer.writerow(newLine)