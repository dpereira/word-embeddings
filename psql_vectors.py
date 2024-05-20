import pgvector.sqlalchemy
import sqlalchemy
import sqlalchemy.ext.declarative



engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:admin@localhost/w2v')
Base = sqlalchemy.ext.declarative.declarative_base()


class Embedding(Base):
    __tablename__ = 'embeddings'

    term = sqlalchemy.Column(sqlalchemy.String(128), primary_key=True)
    vector = sqlalchemy.Column(pgvector.sqlalchemy.Vector(128))

Base.metadata.create_all(engine)


def index():
    import w2v_index
    with sqlalchemy.orm.Session(engine) as session:
        for i, (k, v) in enumerate(w2v_index.index.items()):
            session.add(Embedding(term=k, vector=v))
            if i % 100 == 0:
                session.commit()
        session.commit()
