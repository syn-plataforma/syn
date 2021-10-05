'use strict';


customElements.define('compodoc-menu', class extends HTMLElement {
    constructor() {
        super();
        this.isNormalMode = this.getAttribute('mode') === 'normal';
    }

    connectedCallback() {
        this.render(this.isNormalMode);
    }

    render(isNormalMode) {
        let tp = lithtml.html(`
        <nav>
            <ul class="list">
                <li class="title">
                    <a href="index.html" data-type="index-link">angular-skeleton documentation</a>
                </li>

                <li class="divider"></li>
                ${ isNormalMode ? `<div id="book-search-input" role="search"><input type="text" placeholder="Type to search"></div>` : '' }
                <li class="chapter">
                    <a data-type="chapter-link" href="index.html"><span class="icon ion-ios-home"></span>Getting started</a>
                    <ul class="links">
                        <li class="link">
                            <a href="overview.html" data-type="chapter-link">
                                <span class="icon ion-ios-keypad"></span>Overview
                            </a>
                        </li>
                        <li class="link">
                            <a href="index.html" data-type="chapter-link">
                                <span class="icon ion-ios-paper"></span>README
                            </a>
                        </li>
                                <li class="link">
                                    <a href="dependencies.html" data-type="chapter-link">
                                        <span class="icon ion-ios-list"></span>Dependencies
                                    </a>
                                </li>
                    </ul>
                </li>
                    <li class="chapter modules">
                        <a data-type="chapter-link" href="modules.html">
                            <div class="menu-toggler linked" data-toggle="collapse" ${ isNormalMode ?
                                'data-target="#modules-links"' : 'data-target="#xs-modules-links"' }>
                                <span class="icon ion-ios-archive"></span>
                                <span class="link-name">Modules</span>
                                <span class="icon ion-ios-arrow-down"></span>
                            </div>
                        </a>
                        <ul class="links collapse " ${ isNormalMode ? 'id="modules-links"' : 'id="xs-modules-links"' }>
                            <li class="link">
                                <a href="modules/AppModule.html" data-type="entity-link">AppModule</a>
                                    <li class="chapter inner">
                                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ?
                                            'data-target="#components-links-module-AppModule-0e2d780666b59179b052fd9f7b1dadb7"' : 'data-target="#xs-components-links-module-AppModule-0e2d780666b59179b052fd9f7b1dadb7"' }>
                                            <span class="icon ion-md-cog"></span>
                                            <span>Components</span>
                                            <span class="icon ion-ios-arrow-down"></span>
                                        </div>
                                        <ul class="links collapse" ${ isNormalMode ? 'id="components-links-module-AppModule-0e2d780666b59179b052fd9f7b1dadb7"' :
                                            'id="xs-components-links-module-AppModule-0e2d780666b59179b052fd9f7b1dadb7"' }>
                                            <li class="link">
                                                <a href="components/AppComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">AppComponent</a>
                                            </li>
                                        </ul>
                                    </li>
                                <li class="chapter inner">
                                    <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ?
                                        'data-target="#injectables-links-module-AppModule-0e2d780666b59179b052fd9f7b1dadb7"' : 'data-target="#xs-injectables-links-module-AppModule-0e2d780666b59179b052fd9f7b1dadb7"' }>
                                        <span class="icon ion-md-arrow-round-down"></span>
                                        <span>Injectables</span>
                                        <span class="icon ion-ios-arrow-down"></span>
                                    </div>
                                    <ul class="links collapse" ${ isNormalMode ? 'id="injectables-links-module-AppModule-0e2d780666b59179b052fd9f7b1dadb7"' :
                                        'id="xs-injectables-links-module-AppModule-0e2d780666b59179b052fd9f7b1dadb7"' }>
                                        <li class="link">
                                            <a href="injectables/AuthService.html"
                                                data-type="entity-link" data-context="sub-entity" data-context-id="modules" }>AuthService</a>
                                        </li>
                                        <li class="link">
                                            <a href="injectables/SharingdataService.html"
                                                data-type="entity-link" data-context="sub-entity" data-context-id="modules" }>SharingdataService</a>
                                        </li>
                                        <li class="link">
                                            <a href="injectables/SynApiService.html"
                                                data-type="entity-link" data-context="sub-entity" data-context-id="modules" }>SynApiService</a>
                                        </li>
                                    </ul>
                                </li>
                            </li>
                            <li class="link">
                                <a href="modules/AppRoutingModule.html" data-type="entity-link">AppRoutingModule</a>
                            </li>
                            <li class="link">
                                <a href="modules/DocscModule.html" data-type="entity-link">DocscModule</a>
                                    <li class="chapter inner">
                                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ?
                                            'data-target="#components-links-module-DocscModule-c4ae42344ec53ebe987c0c0b6467a644"' : 'data-target="#xs-components-links-module-DocscModule-c4ae42344ec53ebe987c0c0b6467a644"' }>
                                            <span class="icon ion-md-cog"></span>
                                            <span>Components</span>
                                            <span class="icon ion-ios-arrow-down"></span>
                                        </div>
                                        <ul class="links collapse" ${ isNormalMode ? 'id="components-links-module-DocscModule-c4ae42344ec53ebe987c0c0b6467a644"' :
                                            'id="xs-components-links-module-DocscModule-c4ae42344ec53ebe987c0c0b6467a644"' }>
                                            <li class="link">
                                                <a href="components/DocscComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">DocscComponent</a>
                                            </li>
                                        </ul>
                                    </li>
                            </li>
                            <li class="link">
                                <a href="modules/DocscRoutingModule.html" data-type="entity-link">DocscRoutingModule</a>
                            </li>
                            <li class="link">
                                <a href="modules/IndexModule.html" data-type="entity-link">IndexModule</a>
                                    <li class="chapter inner">
                                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ?
                                            'data-target="#components-links-module-IndexModule-1c5cd570cb88b957f46cf516a4e2e5d6"' : 'data-target="#xs-components-links-module-IndexModule-1c5cd570cb88b957f46cf516a4e2e5d6"' }>
                                            <span class="icon ion-md-cog"></span>
                                            <span>Components</span>
                                            <span class="icon ion-ios-arrow-down"></span>
                                        </div>
                                        <ul class="links collapse" ${ isNormalMode ? 'id="components-links-module-IndexModule-1c5cd570cb88b957f46cf516a4e2e5d6"' :
                                            'id="xs-components-links-module-IndexModule-1c5cd570cb88b957f46cf516a4e2e5d6"' }>
                                            <li class="link">
                                                <a href="components/IndexComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">IndexComponent</a>
                                            </li>
                                        </ul>
                                    </li>
                            </li>
                            <li class="link">
                                <a href="modules/IndexRoutingModule.html" data-type="entity-link">IndexRoutingModule</a>
                            </li>
                            <li class="link">
                                <a href="modules/NotificationModule.html" data-type="entity-link">NotificationModule</a>
                                    <li class="chapter inner">
                                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ?
                                            'data-target="#components-links-module-NotificationModule-9cd0f0ca641221ff0d264dd5ba762c14"' : 'data-target="#xs-components-links-module-NotificationModule-9cd0f0ca641221ff0d264dd5ba762c14"' }>
                                            <span class="icon ion-md-cog"></span>
                                            <span>Components</span>
                                            <span class="icon ion-ios-arrow-down"></span>
                                        </div>
                                        <ul class="links collapse" ${ isNormalMode ? 'id="components-links-module-NotificationModule-9cd0f0ca641221ff0d264dd5ba762c14"' :
                                            'id="xs-components-links-module-NotificationModule-9cd0f0ca641221ff0d264dd5ba762c14"' }>
                                            <li class="link">
                                                <a href="components/NotificationComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">NotificationComponent</a>
                                            </li>
                                        </ul>
                                    </li>
                            </li>
                            <li class="link">
                                <a href="modules/SharedModule.html" data-type="entity-link">SharedModule</a>
                                    <li class="chapter inner">
                                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ?
                                            'data-target="#components-links-module-SharedModule-d00db43eaf227f4f74efec4a09b22f31"' : 'data-target="#xs-components-links-module-SharedModule-d00db43eaf227f4f74efec4a09b22f31"' }>
                                            <span class="icon ion-md-cog"></span>
                                            <span>Components</span>
                                            <span class="icon ion-ios-arrow-down"></span>
                                        </div>
                                        <ul class="links collapse" ${ isNormalMode ? 'id="components-links-module-SharedModule-d00db43eaf227f4f74efec4a09b22f31"' :
                                            'id="xs-components-links-module-SharedModule-d00db43eaf227f4f74efec4a09b22f31"' }>
                                            <li class="link">
                                                <a href="components/FooterComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">FooterComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/NavbarComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">NavbarComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/NotFoundComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">NotFoundComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/TableComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">TableComponent</a>
                                            </li>
                                        </ul>
                                    </li>
                            </li>
                            <li class="link">
                                <a href="modules/SharedRoutingModule.html" data-type="entity-link">SharedRoutingModule</a>
                            </li>
                            <li class="link">
                                <a href="modules/TestTrainingModule.html" data-type="entity-link">TestTrainingModule</a>
                                    <li class="chapter inner">
                                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ?
                                            'data-target="#components-links-module-TestTrainingModule-0c763557a01a91e9754c0a6d11e6d31e"' : 'data-target="#xs-components-links-module-TestTrainingModule-0c763557a01a91e9754c0a6d11e6d31e"' }>
                                            <span class="icon ion-md-cog"></span>
                                            <span>Components</span>
                                            <span class="icon ion-ios-arrow-down"></span>
                                        </div>
                                        <ul class="links collapse" ${ isNormalMode ? 'id="components-links-module-TestTrainingModule-0c763557a01a91e9754c0a6d11e6d31e"' :
                                            'id="xs-components-links-module-TestTrainingModule-0c763557a01a91e9754c0a6d11e6d31e"' }>
                                            <li class="link">
                                                <a href="components/DisplayComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">DisplayComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/InputTestComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">InputTestComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/TestTrainingComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">TestTrainingComponent</a>
                                            </li>
                                        </ul>
                                    </li>
                            </li>
                            <li class="link">
                                <a href="modules/TestTrainingRoutingModule.html" data-type="entity-link">TestTrainingRoutingModule</a>
                            </li>
                            <li class="link">
                                <a href="modules/TourModule.html" data-type="entity-link">TourModule</a>
                                    <li class="chapter inner">
                                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ?
                                            'data-target="#components-links-module-TourModule-2a361fa65aea8b544f9916f71f0b83e3"' : 'data-target="#xs-components-links-module-TourModule-2a361fa65aea8b544f9916f71f0b83e3"' }>
                                            <span class="icon ion-md-cog"></span>
                                            <span>Components</span>
                                            <span class="icon ion-ios-arrow-down"></span>
                                        </div>
                                        <ul class="links collapse" ${ isNormalMode ? 'id="components-links-module-TourModule-2a361fa65aea8b544f9916f71f0b83e3"' :
                                            'id="xs-components-links-module-TourModule-2a361fa65aea8b544f9916f71f0b83e3"' }>
                                            <li class="link">
                                                <a href="components/CodebookComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">CodebookComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/EmbeddingComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">EmbeddingComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/FinalStepComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">FinalStepComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/InputComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">InputComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/NlpComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">NlpComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/TourComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">TourComponent</a>
                                            </li>
                                            <li class="link">
                                                <a href="components/VectorizerComponent.html"
                                                    data-type="entity-link" data-context="sub-entity" data-context-id="modules">VectorizerComponent</a>
                                            </li>
                                        </ul>
                                    </li>
                            </li>
                            <li class="link">
                                <a href="modules/TourRoutingModule.html" data-type="entity-link">TourRoutingModule</a>
                            </li>
                </ul>
                </li>
                        <li class="chapter">
                            <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ? 'data-target="#directives-links"' :
                                'data-target="#xs-directives-links"' }>
                                <span class="icon ion-md-code-working"></span>
                                <span>Directives</span>
                                <span class="icon ion-ios-arrow-down"></span>
                            </div>
                            <ul class="links collapse " ${ isNormalMode ? 'id="directives-links"' : 'id="xs-directives-links"' }>
                                <li class="link">
                                    <a href="directives/DragDropDirective.html" data-type="entity-link">DragDropDirective</a>
                                </li>
                            </ul>
                        </li>
                    <li class="chapter">
                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ? 'data-target="#classes-links"' :
                            'data-target="#xs-classes-links"' }>
                            <span class="icon ion-ios-paper"></span>
                            <span>Classes</span>
                            <span class="icon ion-ios-arrow-down"></span>
                        </div>
                        <ul class="links collapse " ${ isNormalMode ? 'id="classes-links"' : 'id="xs-classes-links"' }>
                            <li class="link">
                                <a href="classes/AppPage.html" data-type="entity-link">AppPage</a>
                            </li>
                            <li class="link">
                                <a href="classes/CodebookRequest.html" data-type="entity-link">CodebookRequest</a>
                            </li>
                            <li class="link">
                                <a href="classes/EmbeddingRequest.html" data-type="entity-link">EmbeddingRequest</a>
                            </li>
                            <li class="link">
                                <a href="classes/IndexToTourModel.html" data-type="entity-link">IndexToTourModel</a>
                            </li>
                            <li class="link">
                                <a href="classes/InfoRowModel.html" data-type="entity-link">InfoRowModel</a>
                            </li>
                            <li class="link">
                                <a href="classes/LoginRequest.html" data-type="entity-link">LoginRequest</a>
                            </li>
                            <li class="link">
                                <a href="classes/MyError.html" data-type="entity-link">MyError</a>
                            </li>
                            <li class="link">
                                <a href="classes/NlpRequest.html" data-type="entity-link">NlpRequest</a>
                            </li>
                            <li class="link">
                                <a href="classes/TableParametersModel.html" data-type="entity-link">TableParametersModel</a>
                            </li>
                            <li class="link">
                                <a href="classes/TableRow.html" data-type="entity-link">TableRow</a>
                            </li>
                            <li class="link">
                                <a href="classes/VectorizerRequest.html" data-type="entity-link">VectorizerRequest</a>
                            </li>
                        </ul>
                    </li>
                        <li class="chapter">
                            <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ? 'data-target="#injectables-links"' :
                                'data-target="#xs-injectables-links"' }>
                                <span class="icon ion-md-arrow-round-down"></span>
                                <span>Injectables</span>
                                <span class="icon ion-ios-arrow-down"></span>
                            </div>
                            <ul class="links collapse " ${ isNormalMode ? 'id="injectables-links"' : 'id="xs-injectables-links"' }>
                                <li class="link">
                                    <a href="injectables/ActionStackEffects.html" data-type="entity-link">ActionStackEffects</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/AuthService.html" data-type="entity-link">AuthService</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/CustomToastrService.html" data-type="entity-link">CustomToastrService</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/EditUserEffects.html" data-type="entity-link">EditUserEffects</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/LocalStorage.html" data-type="entity-link">LocalStorage</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/LocalStorageApi.html" data-type="entity-link">LocalStorageApi</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/LoggerService.html" data-type="entity-link">LoggerService</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/NotificationService.html" data-type="entity-link">NotificationService</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/SharingdataService.html" data-type="entity-link">SharingdataService</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/SpinnerFacade.html" data-type="entity-link">SpinnerFacade</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/SynApiService.html" data-type="entity-link">SynApiService</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/UserApi.html" data-type="entity-link">UserApi</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/UserEffects.html" data-type="entity-link">UserEffects</a>
                                </li>
                                <li class="link">
                                    <a href="injectables/UserFacade.html" data-type="entity-link">UserFacade</a>
                                </li>
                            </ul>
                        </li>
                    <li class="chapter">
                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ? 'data-target="#interceptors-links"' :
                            'data-target="#xs-interceptors-links"' }>
                            <span class="icon ion-ios-swap"></span>
                            <span>Interceptors</span>
                            <span class="icon ion-ios-arrow-down"></span>
                        </div>
                        <ul class="links collapse " ${ isNormalMode ? 'id="interceptors-links"' : 'id="xs-interceptors-links"' }>
                            <li class="link">
                                <a href="interceptors/HttpErrorsResponseInterceptor.html" data-type="entity-link">HttpErrorsResponseInterceptor</a>
                            </li>
                            <li class="link">
                                <a href="interceptors/HttpMethodOverrideInterceptor.html" data-type="entity-link">HttpMethodOverrideInterceptor</a>
                            </li>
                            <li class="link">
                                <a href="interceptors/HttpSwitchUserInterceptor.html" data-type="entity-link">HttpSwitchUserInterceptor</a>
                            </li>
                            <li class="link">
                                <a href="interceptors/TokenInterceptorService.html" data-type="entity-link">TokenInterceptorService</a>
                            </li>
                        </ul>
                    </li>
                    <li class="chapter">
                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ? 'data-target="#guards-links"' :
                            'data-target="#xs-guards-links"' }>
                            <span class="icon ion-ios-lock"></span>
                            <span>Guards</span>
                            <span class="icon ion-ios-arrow-down"></span>
                        </div>
                        <ul class="links collapse " ${ isNormalMode ? 'id="guards-links"' : 'id="xs-guards-links"' }>
                            <li class="link">
                                <a href="guards/AuthGuard.html" data-type="entity-link">AuthGuard</a>
                            </li>
                        </ul>
                    </li>
                    <li class="chapter">
                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ? 'data-target="#interfaces-links"' :
                            'data-target="#xs-interfaces-links"' }>
                            <span class="icon ion-md-information-circle-outline"></span>
                            <span>Interfaces</span>
                            <span class="icon ion-ios-arrow-down"></span>
                        </div>
                        <ul class="links collapse " ${ isNormalMode ? ' id="interfaces-links"' : 'id="xs-interfaces-links"' }>
                            <li class="link">
                                <a href="interfaces/ActionStackState.html" data-type="entity-link">ActionStackState</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/AllStepParameters.html" data-type="entity-link">AllStepParameters</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/AllStepParamsRequest.html" data-type="entity-link">AllStepParamsRequest</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/AllStepsEndpointResponse.html" data-type="entity-link">AllStepsEndpointResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/AllStepsParametersResponse.html" data-type="entity-link">AllStepsParametersResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/AllTasksReponse.html" data-type="entity-link">AllTasksReponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/BasePerson.html" data-type="entity-link">BasePerson</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/CodebookParameters.html" data-type="entity-link">CodebookParameters</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/CodebookParametersReponse.html" data-type="entity-link">CodebookParametersReponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/CodebookResponse.html" data-type="entity-link">CodebookResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/CustomAction.html" data-type="entity-link">CustomAction</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/DatoEjemplo.html" data-type="entity-link">DatoEjemplo</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/DatoEjemplo-1.html" data-type="entity-link">DatoEjemplo</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/EmbeddingResponse.html" data-type="entity-link">EmbeddingResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/EmbeddingsParameters.html" data-type="entity-link">EmbeddingsParameters</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/EmbeddingsParametersResponse.html" data-type="entity-link">EmbeddingsParametersResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/EndPoint.html" data-type="entity-link">EndPoint</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/Fila.html" data-type="entity-link">Fila</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/LoginData.html" data-type="entity-link">LoginData</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/LoginResponse.html" data-type="entity-link">LoginResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/NlpParameters.html" data-type="entity-link">NlpParameters</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/NlpParametersReponse.html" data-type="entity-link">NlpParametersReponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/NlpResponse.html" data-type="entity-link">NlpResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/ObjetoPrueba.html" data-type="entity-link">ObjetoPrueba</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/Person.html" data-type="entity-link">Person</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/PruebaModeloDado.html" data-type="entity-link">PruebaModeloDado</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/PruebaModeloRecibido.html" data-type="entity-link">PruebaModeloRecibido</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/RefreshTokenResponse.html" data-type="entity-link">RefreshTokenResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/SonAction.html" data-type="entity-link">SonAction</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/Spinner.html" data-type="entity-link">Spinner</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/State.html" data-type="entity-link">State</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/Task.html" data-type="entity-link">Task</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/User.html" data-type="entity-link">User</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/UserResponse.html" data-type="entity-link">UserResponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/VectorizerParameters.html" data-type="entity-link">VectorizerParameters</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/VectorizerParametersReponse.html" data-type="entity-link">VectorizerParametersReponse</a>
                            </li>
                            <li class="link">
                                <a href="interfaces/VectorizerResponse.html" data-type="entity-link">VectorizerResponse</a>
                            </li>
                        </ul>
                    </li>
                    <li class="chapter">
                        <div class="simple menu-toggler" data-toggle="collapse" ${ isNormalMode ? 'data-target="#miscellaneous-links"'
                            : 'data-target="#xs-miscellaneous-links"' }>
                            <span class="icon ion-ios-cube"></span>
                            <span>Miscellaneous</span>
                            <span class="icon ion-ios-arrow-down"></span>
                        </div>
                        <ul class="links collapse " ${ isNormalMode ? 'id="miscellaneous-links"' : 'id="xs-miscellaneous-links"' }>
                            <li class="link">
                                <a href="miscellaneous/enumerations.html" data-type="entity-link">Enums</a>
                            </li>
                            <li class="link">
                                <a href="miscellaneous/functions.html" data-type="entity-link">Functions</a>
                            </li>
                            <li class="link">
                                <a href="miscellaneous/variables.html" data-type="entity-link">Variables</a>
                            </li>
                        </ul>
                    </li>
                        <li class="chapter">
                            <a data-type="chapter-link" href="routes.html"><span class="icon ion-ios-git-branch"></span>Routes</a>
                        </li>
                    <li class="chapter">
                        <a data-type="chapter-link" href="coverage.html"><span class="icon ion-ios-stats"></span>Documentation coverage</a>
                    </li>
                    <li class="divider"></li>
                    <li class="copyright">
                        Documentation generated using <a href="https://compodoc.app/" target="_blank">
                            <img data-src="images/compodoc-vectorise.png" class="img-responsive" data-type="compodoc-logo">
                        </a>
                    </li>
            </ul>
        </nav>
        `);
        this.innerHTML = tp.strings;
    }
});