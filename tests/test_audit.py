import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from embedding_claim_audit import phrases
from embedding_claim_audit.audit import audit, evidence, paragraphs, POSITIVE, NEGATIVE, normalize
from embedding_claim_audit.__main__ import main

EXPECTED = {'EMBEDDING': ('embedding', 'embeddings', 'text embedding', 'text embeddings', 'embedding model', 'embedding models', 'generates embeddings', 'generate embeddings', 'generating embeddings', 'embeds text', 'embed text', 'embeds documents', 'embed documents', 'vector embedding', 'vector embeddings', 'semantic embedding', 'semantic embeddings', 'sentence embedding', 'sentence embeddings', 'document embedding', 'document embeddings', 'word embedding', 'word embeddings', 'embedding vector', 'embedding vectors', 'embedding generation', 'creates embeddings', 'create embeddings', 'produces embeddings', 'produce embeddings', 'converts text to vectors', 'converts text into vectors', 'text to vector', 'text to vectors', 'dense vector', 'dense vectors', 'dense representation', 'dense representations', 'vector representation', 'vector representations', 'embedding api', 'embedding service', 'embedding endpoint', 'embeddings api', 'pretrained embedding', 'pretrained embeddings', 'contextual embedding', 'contextual embeddings', 'multilingual embedding', 'multilingual embeddings'), 'NOEMBEDDING': ('no embedding', 'no embeddings', 'does not generate embeddings', 'without embeddings', 'no text embedding', 'does not embed text', 'no embedding model', 'does not support embeddings', 'embedding not supported', 'no vector embedding', 'does not use embeddings', 'no dense vector', 'does not create embeddings', 'no semantic embedding', 'does not produce embeddings', 'keyword only', 'keyword search only', 'no sentence embedding', 'does not convert text to vectors', 'tf-idf only', 'sparse only', 'no word embedding', 'does not embed documents', 'one-hot only'), 'GENERIC_NAMES': ('embed_text', 'embed_texts', 'embed_document', 'embed_documents', 'get_embedding', 'get_embeddings', 'create_embedding', 'create_embeddings', 'generate_embedding', 'generate_embeddings', 'encode_text', 'encode_texts', 'vectorize_text', 'text_to_embedding', 'text_to_vector', 'embed', 'embed_batch', 'embed_documents_batch', 'compute_embedding', 'embedding', 'encode_document', 'encode_documents', 'embed_sentence', 'embed_sentences'), 'SENTENCE_CONSTRUCTIONS': ('SentenceTransformer',), 'SENTENCE_OPERATIONS': ('encode',), 'OPENAI_CONSTRUCTIONS': ('OpenAI', 'AsyncOpenAI'), 'OPENAI_OPERATIONS': ('embeddings.create', 'Embedding.create'), 'COHERE_CONSTRUCTIONS': ('Client', 'ClientV2', 'AsyncClient', 'AsyncClientV2'), 'COHERE_OPERATIONS': ('embed',), 'PIPELINE_CONSTRUCTIONS': ('pipeline',), 'PIPELINE_TASK': ('feature-extraction',), 'LANGCHAIN_CONSTRUCTIONS': ('OpenAIEmbeddings', 'HuggingFaceEmbeddings', 'CohereEmbeddings', 'GoogleGenerativeAIEmbeddings', 'AzureOpenAIEmbeddings', 'OllamaEmbeddings', 'BedrockEmbeddings', 'VertexAIEmbeddings', 'VoyageAIEmbeddings', 'MistralAIEmbeddings'), 'LANGCHAIN_OPERATIONS': ('embed_query', 'embed_documents'), 'LLAMA_CONSTRUCTIONS': ('OpenAIEmbedding', 'HuggingFaceEmbedding', 'CohereEmbedding'), 'LLAMA_OPERATIONS': ('get_text_embedding', 'get_text_embeddings', 'get_text_embedding_batch', 'get_query_embedding', 'aget_text_embedding', 'aget_text_embeddings', 'aget_text_embedding_batch', 'aget_query_embedding')}

class AuditTests(unittest.TestCase):
    def fixture(self, doc='We generate embeddings.', files=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / 'README.md').write_text(doc)
        for name, source in (files or {}).items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        return root

    def cli(self, root, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(['audit', str(root), *args])
        return code, output.getvalue()

    def test_phrase_oracle(self):
        for key, expected in EXPECTED.items():
            actual = getattr(phrases, key)
            self.assertEqual(tuple(map(normalize, actual)), tuple(map(normalize, expected)))
            self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(len(phrases.EMBEDDING), 50)
        self.assertEqual(len(phrases.NOEMBEDDING), 24)
        self.assertEqual(len(phrases.GENERIC_NAMES), 24)

    def test_good(self):
        root = self.fixture(files={'app.py': 'x.embed_text("hi")'})
        self.assertEqual(self.cli(root)[0], 0)
        self.assertEqual(audit(root)['verdict'], 'PASS')

    def test_bad(self):
        root = self.fixture(files={'app.py': 'pass'})
        code, output = self.cli(root)
        self.assertEqual(code, 2)
        self.assertIn('documented-embedding-claim-false HIGH', output)

    def test_test_only(self):
        root = self.fixture(files={'tests/test_app.py': 'x.embed_text("hi")'})
        self.assertEqual(self.cli(root)[0], 0)
        self.assertEqual(self.cli(root, '--fail-on', 'medium')[0], 2)
        self.assertEqual(audit(root)['findings'][0]['severity'], 'MEDIUM')

    def test_denial(self):
        root = self.fixture('No embeddings.', {'app.py': 'x.embed_text("hi")'})
        code, output = self.cli(root)
        self.assertEqual(code, 2)
        self.assertIn('documented-no-embedding-claim-false HIGH', output)
        self.assertEqual(len(audit(root)['findings']), 1)

    def test_no_claim(self):
        self.assertEqual(audit(self.fixture('A useful tool.'))['verdict'], 'PASS')

    def test_no_source_asymmetry(self):
        self.assertEqual(self.cli(self.fixture())[0], 2)
        self.assertEqual(self.cli(self.fixture('No embeddings.'))[0], 0)

    def test_fences(self):
        for fence in ('```', '~~~', '   ````'):
            self.assertEqual(audit(self.fixture(fence+'text\ngenerate embeddings\n'+fence.strip()))['verdict'], 'PASS')

    def test_inline(self):
        self.assertEqual(self.cli(self.fixture('We `generate embeddings`.'))[0], 2)

    def test_normalization(self):
        self.assertTrue(NEGATIVE.search(normalize("doesn't generate embeddings")))
        self.assertTrue(POSITIVE.search(normalize('TEXT-TO-VECTORS')))
        self.assertFalse(POSITIVE.search('reembeddings'))
        self.assertEqual(NEGATIVE.sub('', normalize('does not generate embeddings')), '')
        self.assertEqual(paragraphs('first\nline\n\nsecond'), ['first line', 'second'])

    def test_module_boundary(self):
        root = self.fixture(files={'a.py': 'from sentence_transformers import SentenceTransformer\nm=SentenceTransformer()', 'b.py': 'm.encode("hi")'})
        self.assertEqual(self.cli(root)[0], 2)

    def test_exclusions(self):
        for name in ('fixtures/a.py', 'docs/a.py', 'examples/a.py'):
            self.assertEqual(audit(self.fixture(files={name:'x.embed_text()'}))['findings'][0]['severity'], 'HIGH')
        for name in ('tests/a.py', 'test_a.py', 'a_test.py'):
            self.assertEqual(audit(self.fixture(files={name:'x.embed_text()'}))['findings'][0]['severity'], 'MEDIUM')

    def test_symlinks(self):
        root = self.fixture()
        other = self.fixture(files={'app.py':'x.embed_text()'})
        (root/'app.py').symlink_to(other/'app.py')
        (root/'linked').symlink_to(other, target_is_directory=True)
        self.assertEqual(self.cli(root)[0], 2)

    def test_docs_selection_and_ignore(self):
        root = self.fixture('Safe.', {'nested/docs/page.md':'embedding', 'other.md':'embedding', 'vendor/a.py':'x.embed_text()'})
        self.assertEqual(self.cli(root)[0], 0)
        self.assertEqual(self.cli(root, '--ignore', 'vendor/')[0], 2)
        self.assertEqual(self.cli(root, '--ignore', 'vendor/', '--ignore', 'nested/')[0], 0)

    def test_json(self):
        code, output = self.cli(self.fixture(), '--json')
        result = json.loads(output)
        self.assertEqual(code, 2)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(result['findings'][0]['message'])

    def test_usage_errors(self):
        for args in ([], ['audit', '--fail-on', 'MEDIUM'], ['audit', '--ignore'], ['unknown']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(args)
            self.assertEqual(caught.exception.code, 1)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.cli('/does/not/exist')[0], 1)

    def test_version(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as caught:
            main(['--version'])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(stream.getvalue().strip(), '0.1.0')

    def test_entry_points(self):
        result = subprocess.run([sys.executable, '-m', 'embedding_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '0.1.0')
        import shutil
        console = shutil.which('embedding-claim-audit')
        if console:
            result = subprocess.run([console, '--version'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), '0.1.0')

    def test_route_sentence(self):
        self.assertTrue(evidence('import sentence_transformers as st\nm=st.SentenceTransformer()\nm.encode([])'))

    def test_route_sentence_chain(self):
        self.assertTrue(evidence('import sentence_transformers as st\nst.SentenceTransformer().encode([])'))

    def test_route_sentence_from(self):
        self.assertTrue(evidence('from sentence_transformers import SentenceTransformer\nm=SentenceTransformer()\nm.encode([])'))

    def test_route_openai(self):
        self.assertTrue(evidence('import openai\nc=openai.OpenAI()\nc.embeddings.create()'))

    def test_route_openai_chain(self):
        self.assertTrue(evidence('from openai import OpenAI\nOpenAI().embeddings.create()'))

    def test_route_openai_legacy(self):
        self.assertTrue(evidence('import openai\nopenai.Embedding.create()'))

    def test_route_openai_legacy_from(self):
        self.assertTrue(evidence('from openai import Embedding\nEmbedding.create()'))

    def test_route_openai_async(self):
        self.assertTrue(evidence('from openai import AsyncOpenAI\nc=AsyncOpenAI()\nc.embeddings.create()'))

    def test_route_cohere(self):
        self.assertTrue(evidence('import cohere\nc=cohere.Client()\nc.embed()'))

    def test_route_cohere_v2(self):
        self.assertTrue(evidence('from cohere import ClientV2\nc=ClientV2()\nc.embed()'))

    def test_route_cohere_chain(self):
        self.assertTrue(evidence('import cohere\ncohere.Client().embed()'))

    def test_route_pipeline(self):
        self.assertTrue(evidence('from transformers import pipeline\np=pipeline("feature-extraction")\np("hi")'))

    def test_route_pipeline_kw(self):
        self.assertTrue(evidence('from transformers import pipeline\np=pipeline(task="feature-extraction")\np("hi")'))

    def test_route_pipeline_chain(self):
        self.assertTrue(evidence('from transformers import pipeline\npipeline("feature-extraction")("hi")'))

    def test_route_langchain(self):
        self.assertTrue(evidence('from langchain_openai import OpenAIEmbeddings\ne=OpenAIEmbeddings()\ne.embed_query("hi")'))

    def test_route_langchain_hf(self):
        self.assertTrue(evidence('from langchain.embeddings import HuggingFaceEmbeddings\ne=HuggingFaceEmbeddings()\ne.embed_documents([])'))

    def test_route_langchain_chain(self):
        self.assertTrue(evidence('from langchain_openai import OpenAIEmbeddings\nOpenAIEmbeddings().embed_query("hi")'))

    def test_route_llama(self):
        self.assertTrue(evidence('from llama_index.embeddings.openai import OpenAIEmbedding\ne=OpenAIEmbedding()\ne.get_text_embedding("hi")'))

    def test_route_llama_query(self):
        self.assertTrue(evidence('from llama_index.embeddings.openai import OpenAIEmbedding\ne=OpenAIEmbedding()\ne.get_query_embedding("hi")'))

    def test_route_llama_batch(self):
        self.assertTrue(evidence('from llama_index.embeddings.openai import OpenAIEmbedding\ne=OpenAIEmbedding()\ne.get_text_embedding_batch([])'))

    def test_route_llama_chain(self):
        self.assertTrue(evidence('from llama_index.embeddings.openai import OpenAIEmbedding\nOpenAIEmbedding().get_text_embedding("hi")'))

    def test_route_generic_from(self):
        self.assertTrue(evidence('from custom import embed_text\nembed_text("hi")'))

    def test_route_alias(self):
        self.assertTrue(evidence('from sentence_transformers import SentenceTransformer as ST\na=ST()\nb=a\nb.encode([])'))

    def test_route_annassign(self):
        self.assertTrue(evidence('from sentence_transformers import SentenceTransformer\na: object=SentenceTransformer()\na.encode([])'))

    def test_route_with(self):
        self.assertTrue(evidence('from openai import OpenAI\nwith OpenAI() as c:\n c.embeddings.create()'))

    def test_route_asyncwith(self):
        self.assertTrue(evidence('from openai import AsyncOpenAI\nasync def run():\n async with AsyncOpenAI() as c:\n  await c.embeddings.create()'))

    def test_route_generic_alias(self):
        self.assertTrue(evidence('from custom import embed_text as operation\noperation()'))

    def test_route_generic_embed_text(self):
        self.assertTrue(evidence('x.embed_text()'))

    def test_route_generic_embed_texts(self):
        self.assertTrue(evidence('x.embed_texts()'))

    def test_route_generic_embed_document(self):
        self.assertTrue(evidence('x.embed_document()'))

    def test_route_generic_embed_documents(self):
        self.assertTrue(evidence('x.embed_documents()'))

    def test_route_generic_get_embedding(self):
        self.assertTrue(evidence('x.get_embedding()'))

    def test_route_generic_get_embeddings(self):
        self.assertTrue(evidence('x.get_embeddings()'))

    def test_route_generic_create_embedding(self):
        self.assertTrue(evidence('x.create_embedding()'))

    def test_route_generic_create_embeddings(self):
        self.assertTrue(evidence('x.create_embeddings()'))

    def test_route_generic_generate_embedding(self):
        self.assertTrue(evidence('x.generate_embedding()'))

    def test_route_generic_generate_embeddings(self):
        self.assertTrue(evidence('x.generate_embeddings()'))

    def test_route_generic_encode_text(self):
        self.assertTrue(evidence('x.encode_text()'))

    def test_route_generic_encode_texts(self):
        self.assertTrue(evidence('x.encode_texts()'))

    def test_route_generic_vectorize_text(self):
        self.assertTrue(evidence('x.vectorize_text()'))

    def test_route_generic_text_to_embedding(self):
        self.assertTrue(evidence('x.text_to_embedding()'))

    def test_route_generic_text_to_vector(self):
        self.assertTrue(evidence('x.text_to_vector()'))

    def test_route_generic_embed(self):
        self.assertTrue(evidence('x.embed()'))

    def test_route_generic_embed_batch(self):
        self.assertTrue(evidence('x.embed_batch()'))

    def test_route_generic_embed_documents_batch(self):
        self.assertTrue(evidence('x.embed_documents_batch()'))

    def test_route_generic_compute_embedding(self):
        self.assertTrue(evidence('x.compute_embedding()'))

    def test_route_generic_embedding(self):
        self.assertTrue(evidence('x.embedding()'))

    def test_route_generic_encode_document(self):
        self.assertTrue(evidence('x.encode_document()'))

    def test_route_generic_encode_documents(self):
        self.assertTrue(evidence('x.encode_documents()'))

    def test_route_generic_embed_sentence(self):
        self.assertTrue(evidence('x.embed_sentence()'))

    def test_route_generic_embed_sentences(self):
        self.assertTrue(evidence('x.embed_sentences()'))

    def test_route_cohere_Client(self):
        self.assertTrue(evidence('from cohere import Client\nc=Client()\nc.embed()'))

    def test_route_cohere_ClientV2(self):
        self.assertTrue(evidence('from cohere import ClientV2\nc=ClientV2()\nc.embed()'))

    def test_route_cohere_AsyncClient(self):
        self.assertTrue(evidence('from cohere import AsyncClient\nc=AsyncClient()\nc.embed()'))

    def test_route_cohere_AsyncClientV2(self):
        self.assertTrue(evidence('from cohere import AsyncClientV2\nc=AsyncClientV2()\nc.embed()'))

    def test_route_langchain_OpenAIEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import OpenAIEmbeddings\nc=OpenAIEmbeddings()\nc.embed_query()'))

    def test_route_langchain_HuggingFaceEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import HuggingFaceEmbeddings\nc=HuggingFaceEmbeddings()\nc.embed_query()'))

    def test_route_langchain_CohereEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import CohereEmbeddings\nc=CohereEmbeddings()\nc.embed_query()'))

    def test_route_langchain_GoogleGenerativeAIEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import GoogleGenerativeAIEmbeddings\nc=GoogleGenerativeAIEmbeddings()\nc.embed_query()'))

    def test_route_langchain_AzureOpenAIEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import AzureOpenAIEmbeddings\nc=AzureOpenAIEmbeddings()\nc.embed_query()'))

    def test_route_langchain_OllamaEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import OllamaEmbeddings\nc=OllamaEmbeddings()\nc.embed_query()'))

    def test_route_langchain_BedrockEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import BedrockEmbeddings\nc=BedrockEmbeddings()\nc.embed_query()'))

    def test_route_langchain_VertexAIEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import VertexAIEmbeddings\nc=VertexAIEmbeddings()\nc.embed_query()'))

    def test_route_langchain_VoyageAIEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import VoyageAIEmbeddings\nc=VoyageAIEmbeddings()\nc.embed_query()'))

    def test_route_langchain_MistralAIEmbeddings(self):
        self.assertTrue(evidence('from langchain.embeddings import MistralAIEmbeddings\nc=MistralAIEmbeddings()\nc.embed_query()'))

    def test_route_llama_get_text_embedding(self):
        self.assertTrue(evidence('from llama_index import OpenAIEmbedding\nc=OpenAIEmbedding()\nc.get_text_embedding()'))

    def test_route_llama_get_text_embeddings(self):
        self.assertTrue(evidence('from llama_index import OpenAIEmbedding\nc=OpenAIEmbedding()\nc.get_text_embeddings()'))

    def test_route_llama_get_text_embedding_batch(self):
        self.assertTrue(evidence('from llama_index import OpenAIEmbedding\nc=OpenAIEmbedding()\nc.get_text_embedding_batch()'))

    def test_route_llama_get_query_embedding(self):
        self.assertTrue(evidence('from llama_index import OpenAIEmbedding\nc=OpenAIEmbedding()\nc.get_query_embedding()'))

    def test_route_llama_aget_text_embedding(self):
        self.assertTrue(evidence('from llama_index import OpenAIEmbedding\nc=OpenAIEmbedding()\nc.aget_text_embedding()'))

    def test_route_llama_aget_text_embeddings(self):
        self.assertTrue(evidence('from llama_index import OpenAIEmbedding\nc=OpenAIEmbedding()\nc.aget_text_embeddings()'))

    def test_route_llama_aget_text_embedding_batch(self):
        self.assertTrue(evidence('from llama_index import OpenAIEmbedding\nc=OpenAIEmbedding()\nc.aget_text_embedding_batch()'))

    def test_route_llama_aget_query_embedding(self):
        self.assertTrue(evidence('from llama_index import OpenAIEmbedding\nc=OpenAIEmbedding()\nc.aget_query_embedding()'))

    def test_not_evidence_bare_import(self):
        self.assertFalse(evidence('import sentence_transformers, openai, cohere, transformers, langchain, llama_index'))

    def test_not_evidence_lone_sentence(self):
        self.assertFalse(evidence('from sentence_transformers import SentenceTransformer\nm=SentenceTransformer()'))

    def test_not_evidence_lone_encode(self):
        self.assertFalse(evidence('import sentence_transformers\nx.encode()'))

    def test_not_evidence_lone_openai(self):
        self.assertFalse(evidence('from openai import OpenAI\nc=OpenAI()'))

    def test_not_evidence_lone_create(self):
        self.assertFalse(evidence('import openai\nx.embeddings.create()'))

    def test_not_evidence_lone_cohere(self):
        self.assertFalse(evidence('import cohere\nc=cohere.Client()'))

    def test_not_evidence_lone_embed(self):
        self.assertFalse(evidence('import cohere\nx.embed()'))

    def test_not_evidence_lone_pipeline(self):
        self.assertFalse(evidence('from transformers import pipeline\np=pipeline("feature-extraction")'))

    def test_not_evidence_wrong_pipeline(self):
        self.assertFalse(evidence('from transformers import pipeline\np=pipeline("text-classification")\np("hi")'))

    def test_not_evidence_lone_langchain(self):
        self.assertFalse(evidence('import langchain\nx.embed_query()'))

    def test_not_evidence_lone_documents(self):
        self.assertFalse(evidence('import langchain\nx.embed_documents()'))

    def test_not_evidence_lone_llama(self):
        self.assertFalse(evidence('from llama_index import OpenAIEmbedding\nx=OpenAIEmbedding()'))

    def test_not_evidence_lone_text(self):
        self.assertFalse(evidence('import llama_index\nx.get_text_embedding()'))

    def test_not_evidence_layer(self):
        self.assertFalse(evidence('from torch import nn\nx=nn.Embedding(10,5)'))

    def test_not_evidence_tokenize(self):
        self.assertFalse(evidence('x.tokenize("hi")'))

    def test_not_evidence_tfidf(self):
        self.assertFalse(evidence('TfidfVectorizer().fit_transform([])'))

    def test_not_evidence_search(self):
        self.assertFalse(evidence('x.similarity_search("hi")'))

    def test_not_evidence_retrieval(self):
        self.assertFalse(evidence('x.retrieve("hi")'))

    def test_not_evidence_generation(self):
        self.assertFalse(evidence('x.generate("hi")'))

    def test_not_evidence_answer(self):
        self.assertFalse(evidence('from transformers import pipeline\np=pipeline("question-answering")\np("hi")'))

    def test_not_evidence_classify(self):
        self.assertFalse(evidence('x.classify("hi")'))

    def test_not_evidence_finetune(self):
        self.assertFalse(evidence('trainer.train()'))

    def test_not_evidence_image(self):
        self.assertFalse(evidence('clip.encode_image(img)'))

    def test_not_evidence_image_pipeline(self):
        self.assertFalse(evidence('from transformers import pipeline\np=pipeline("image-feature-extraction")\np(img)'))

    def test_not_evidence_subprocess(self):
        self.assertFalse(evidence('import subprocess\nsubprocess.run(["embedding-cli"])'))

    def test_not_evidence_shadow(self):
        self.assertFalse(evidence('from sentence_transformers import SentenceTransformer\ndef SentenceTransformer(): pass\nx=SentenceTransformer()\nx.encode()'))

    def test_not_evidence_shadow_generic(self):
        self.assertFalse(evidence('class X:\n def embed_text(self): pass\nx.embed_text()'))

    def test_not_evidence_shadow_alias(self):
        self.assertFalse(evidence('from custom import embed_text as op\ndef op(): pass\nop()'))

    def test_not_evidence_bare_generic(self):
        self.assertFalse(evidence('embed_text()'))

    def test_not_evidence_bare_reference(self):
        self.assertFalse(evidence('x.embed_text'))

    def test_not_evidence_string(self):
        self.assertFalse(evidence('"x.embed_text()"'))

    def test_not_evidence_comment(self):
        self.assertFalse(evidence('# x.embed_text()'))

    def test_not_evidence_syntax(self):
        self.assertFalse(evidence('x.embed_text('))

    def test_not_evidence_word2vec(self):
        self.assertFalse(evidence('Word2Vec([])'))

    def test_not_evidence_wrong_binding(self):
        self.assertFalse(evidence('from sentence_transformers import SentenceTransformer\nx=Other()\nx.encode()'))

    def test_not_evidence_no_gate(self):
        self.assertFalse(evidence('x=SentenceTransformer()\nx.encode()'))

