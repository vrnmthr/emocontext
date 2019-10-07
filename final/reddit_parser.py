import csv
import os
import sqlite3

body_index = 0
parent_id_index = 10
id_index = 15

c = sqlite3.connect('comments.db')
c.execute('''create table comments (id text, parent_id text, comment text, primary key(id))''')

total_count = 0
num_epochs = 1

directory = os.fsencode("data")
for f in os.listdir(directory):
    fname = os.fsdecode(f)
    print(fname)
    with open("data/" + fname, "r", encoding="utf-8") as fin:
        next(fin)
        reader = csv.reader(fin, delimiter=",")
        for i, line in enumerate(reader):
            id = line[id_index]
            parent_id = line[parent_id_index].split('_')[1]
            body = line[body_index]

            c.execute("insert into comments values (?, ?, ?)", (str(id), str(parent_id), body))
            total_count += 1

            if (total_count % 100000 == 0):
                print(total_count)

        
print("done loading db!")

num_output = 0
all_rows = c.cursor()
all_rows.execute('select * from comments')

with open("reddit_cleaned.csv", "a", encoding="utf-8") as fout:
    writer = csv.writer(fout, delimiter=",")
    for row in all_rows:
        id = row[0]
        parent_id = row[1]
        body = row[2]

        if len(body.split(' ')) > 15:
            continue

        parent = c.cursor().execute("select * from comments where id=?", (parent_id,)).fetchone()
        if parent is not None:
            parent_id = parent[0]
            grandparent_id = parent[1]
            parent_comment = parent[2]
            if len(parent_comment.split(' ')) > 15:
                continue

            grandparent = c.cursor().execute("select * from comments where id=?", (grandparent_id,)).fetchone()
            if grandparent is not None:
                grandparent_id = grandparent[0]
                grandparent_comment = grandparent[2]
                if len(grandparent_comment.split(' ')) > 15:
                    continue

                num_output += 1

                writer.writerow([grandparent_comment, parent_comment, body])
                if num_output % 100 == 0:
                    print(num_output)

print("done!!")