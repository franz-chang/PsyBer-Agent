import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';
import matrixPoisonPaperPdf from '@site/static/papers/self-evolving-matrix-poison-agent-on-recommender-system.pdf';
import methodFigure from '@site/static/img/method-figure.jpg';

const paperList = [
  {
    id: '01',
    title:
      'Goal-guided Generative Prompt Injection Attack on Large Language Models',
    source: 'IEEE Xplore',
    url: 'https://ieeexplore.ieee.org/abstract/document/10884369',
    description:
      "Abstract: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representations x and x' of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on x and x'. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.",
  },
  {
    id: '02',
    title: 'Target-driven Attack for Large Language Models',
    source: 'IOS Press',
    url: 'https://ebooks.iospress.nl/doi/10.3233/FAIA240685',
    description:
      "Abstract: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.",
  },
  {
    id: '03',
    title:
      'Semantic-Preserving Prompt Hijacking: A Black-Box Adversarial Attack on Auto-Prompt Optimization',
    source: 'XJTLU Scholar',
    url: 'https://scholar.xjtlu.edu.cn/en/publications/semantic-preserving-prompt-hijacking-a-black-box-adversarial-atta/',
    description:
      'Abstract: LLMs increasingly integrate auto-suggestion optimization modules, enabling them to rewrite and display user input before generating the final response. While this design aims to enhance transparency and trust, its process of autonomously selecting a single "best" result from multiple candidate solutions allows attackers to hijack this optimization process by inducing subtle, imperceptible semantic shifts. To address this, we propose a semantic preservation hijacking attack method based on black-box conditions - Adaptive Greedy Local Search. This method hierarchically decomposes the input text, masks key language units, and dynamically adjusts candidate replacement words at predefined semantic checkpoints. This maximizes the deviation between the model output and the original intent while strictly maintaining semantic similarity to the original text. Experimental results on commercial and open-source LLM demonstrate that, under the same semantic similarity constraints, this method achieves a higher attack success rate than existing attack methods in over 2400 test cases.',
  },
  {
    id: '04',
    title:
      'Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs',
    source: 'arXiv',
    url: 'https://arxiv.org/abs/2506.17231',
    description:
      "Abstract: As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce Adversarial Prompt Distillation, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations.",
  },
  {
    id: '05',
    title: 'Self-Evolving Matrix-Poison Agent on Recommender System',
    source: 'Attached PDF',
    url: matrixPoisonPaperPdf,
    local: true,
    description:
      'Abstract: Recommender systems are increasingly vulnerable to poisoning attacks, where malicious user data is injected to bias collaborative filtering models. Existing methods typically focus on single-item promotion on individual platforms, overlooking category-level attacks that exploit coordinated multi-account operations to amplify influence and enhance stealth. To address this, we propose Matrix-Poison, a self-evolving LLM-driven agent framework inspired by real-world Matrix Account Advertising. Matrix-Poison overcomes four key hurdles: vulnerability discovery, adaptive black-box planning, autonomous account synergism, and self-correcting stealth. Through iterative reconnaissance and strategy refinement, the agent balances attack impact with concealment. Experiments on real-world datasets demonstrate that Matrix-Poison significantly outperforms state-of-the-art baselines and exhibits strong cross-market portability.',
  },
  {
    id: '06',
    title:
      'AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models',
    source: 'ACM Digital Library',
    url: 'https://dl.acm.org/doi/abs/10.1145/3748239.3748242',
    description:
      "Abstract: Jailbreak attacks represent one of the most sophisticated threats to the security of large language models (LLMs). To deal with such risks, we introduce an innovative framework that can help evaluate the effectiveness of jailbreak attacks on LLMs. Unlike traditional binary evaluations focusing solely on the robustness of LLMs, our method assesses the attacking prompts' effectiveness. We present two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework uses a scoring range from 0 to 1, offering unique perspectives and allowing for the assessment of attack effectiveness in different scenarios. Additionally, we develop a comprehensive ground truth dataset specifically tailored for jailbreak prompts. This dataset is a crucial benchmark for our current study and provides a foundational resource for future research. By comparing with traditional evaluation methods, our study shows that the current results align with baseline metrics while offering a more nuanced and fine-grained assessment. It also helps identify potentially harmful attack prompts that might appear harmless in traditional evaluations. Overall, our work establishes a solid foundation for assessing a broader range of attack prompts in prompt injection.",
  },
  {
    id: '07',
    title: 'Large Vision-Language Model Security: A Survey',
    source: 'Springer',
    url: 'https://link.springer.com/chapter/10.1007/978-981-96-0151-6_1',
    description:
      'Abstract: In the domain of Large Vision-Language Models (LVLMs), securing these models has emerged as a critical issue for both researchers and practitioners. In this paper, we highlight and analyze the security-related issues of LVLMs, with a special emphasis on the reliability challenges in practical deployments. We begin by reviewing recent studies on threats like jailbreak and backdoor attacks, alongside discussing the potential countermeasures implemented to mitigate these risks. Additionally, we touch on real-world application problems, such as hallucinations and privacy leakages, as well as the ethical and legal related researches around them. We also outline the shortcomings observed in current studies and discuss directions for future research, with the aim of promoting LVLMs towards a safer direction.',
  },
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <p className={styles.kicker}>Project Homepage</p>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Distribution-guided adversarial attack literature homepage for LLM and LVLM security.">
      <HomepageHeader />
      <main className={styles.mainContent}>
        <section className={styles.overview}>
          <div className="container">
            <Heading as="h2">Collection Scope</Heading>
            <p>
              This page tracks a focused sequence of seven papers around prompt
              injection, jailbreak transfer, adversarial optimization, and
              attack evaluation for large language models and vision-language
              models.
            </p>
          </div>
        </section>
        <section id="paper-list" className={styles.paperSection}>
          <div className="container">
            <Heading as="h2">Paper List</Heading>
            <div className={styles.paperList}>
              {paperList.map((paper) => (
                <article key={paper.id} className={styles.paperCard}>
                  <div className={styles.paperMeta}>
                    <div className={styles.paperTop}>
                      <span className={styles.paperId}>Paper {paper.id}</span>
                      <span className={styles.paperSource}>{paper.source}</span>
                    </div>
                    <Heading as="h3" className={styles.paperTitle}>
                      {paper.title}
                    </Heading>
                    <p className={styles.paperDesc}>{paper.description}</p>
                    <a
                      className={styles.paperLink}
                      href={paper.url}
                      target={!paper.local ? '_blank' : undefined}
                      rel={!paper.local ? 'noreferrer noopener' : undefined}>
                      View paper
                    </a>
                  </div>
                  <figure className={styles.methodFigure}>
                    <img src={methodFigure} alt={`Method figure for paper ${paper.id}`} />
                  </figure>
                </article>
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
