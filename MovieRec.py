from flask import Flask, render_template, current_app, request
from werkzeug.contrib.cache import SimpleCache
import cPickle
import cf_sgd
import preprocess

app = Flask(__name__)
cache = SimpleCache()


@app.route('/')
def hello_world():
    return render_template("user.html")


@app.route('/user/<int:user_id>')
def user(user_id):
    print user_id
    cf = cache.get("cf")
    if cf is None:
        with open("/Users/shawn/Code/PyCharmWorkspace/MovieRec/cf.pkl","r") as f:
            # cf = cPickle.load(f)
            seened_all, res_all = cPickle.load(f)
    seened_movie = seened_all[user_id]
    res = res_all[user_id]

    return render_template("user.html", seened_movie=seened_movie, res=res)


@app.errorhandler(404)
def not_found(error):
    return error

@app.route("/add/<int:user_id>", methods=["GET"])
def add(user_id):
    movie_name = cf_sgd.load_data()
    return render_template("add.html", user_id=user_id, moviename=movie_name.values(), movieId=range(len(movie_name.keys())))

@app.route("/submit/<int:user_id>", methods=["GET", "POST"])
def submit(user_id):
    form = request.form
    movie_name = cf_sgd.load_data()
    adding = []

    for i in range(len(movie_name.keys())):
        a = form.get(str(i))
        mid = movie_name.keys()[i]
        if a != None and a != "" :
            print a, mid, movie_name[mid], type(a)
            try:
                a = eval(a)
                # mid =eval(mid)
            except Exception,e:
                print e
                continue
            adding.append([user_id, mid, a])
    print adding
    cf_sgd.add_rating(adding)
    return render_template("submit.html")


if __name__ == '__main__':
    # preprocess.simple_bar()

    # cf = cf_sgd.init()
    # cache.set("cf",cf)

    app.run(port=2355, debug=True, threaded=True)

